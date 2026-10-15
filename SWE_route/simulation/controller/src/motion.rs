//! Joint-limited motion. Keep the physics state separate from these requested poses.

use crate::gestures::Gesture;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, VecDeque};

pub type Pose = BTreeMap<String, f64>;

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub struct JointLimit {
    pub lower: f64,
    pub upper: f64,
    pub velocity: f64,
}

impl JointLimit {
    pub fn clamp(&self, value: f64) -> Result<f64, String> {
        if !value.is_finite() {
            return Err("Joint targets must be finite radians".into());
        }
        Ok(value.max(self.lower).min(self.upper))
    }
}

#[derive(Clone, Debug)]
struct Segment {
    targets: Pose,
    seconds: f64,
    hold: bool,
}

/// Single-owner controller with the same cubic easing and segment semantics as Python.
#[derive(Clone, Debug)]
pub struct MotionController {
    pub limits: BTreeMap<String, JointLimit>,
    pub positions: Pose,
    pub targets: Pose,
    pub activity: String,
    start: Pose,
    elapsed: f64,
    duration: f64,
    waypoints: VecDeque<Segment>,
    speed: f64,
}

impl MotionController {
    pub fn new(limits: BTreeMap<String, JointLimit>) -> Result<Self, String> {
        if limits.is_empty() {
            return Err("The model has no revolute joints".into());
        }
        for (name, limit) in &limits {
            if ![limit.lower, limit.upper, limit.velocity]
                .iter()
                .all(|x| x.is_finite())
                || limit.lower > limit.upper
                || limit.velocity <= 0.0
            {
                return Err(format!("Invalid limits for {name}"));
            }
        }
        let positions: Pose = limits
            .iter()
            .map(|(name, limit)| Ok((name.clone(), limit.clamp(0.0)?)))
            .collect::<Result<_, String>>()?;
        Ok(Self {
            limits,
            targets: positions.clone(),
            start: positions.clone(),
            positions,
            activity: "Ready".into(),
            elapsed: 0.0,
            duration: 0.0,
            waypoints: VecDeque::new(),
            speed: 1.0,
        })
    }

    pub fn speed(&self) -> f64 {
        self.speed
    }

    pub fn set_speed(&mut self, speed: f64) -> Result<(), String> {
        if !speed.is_finite() || !(0.0 < speed && speed <= 1.0) {
            return Err("Playback speed must be greater than zero and at most one".into());
        }
        self.speed = speed;
        Ok(())
    }

    pub fn moving(&self) -> bool {
        self.elapsed < self.duration || !self.waypoints.is_empty()
    }

    fn begin(&mut self, segment: Segment) {
        self.start = self.positions.clone();
        self.targets = segment.targets;
        self.elapsed = 0.0;
        self.duration = if segment.hold {
            segment.seconds
        } else {
            self.limits
                .iter()
                .fold(0.25_f64.max(segment.seconds), |duration, (name, limit)| {
                    duration.max(
                        1.5 * (self.targets[name] - self.positions[name]).abs() / limit.velocity,
                    )
                })
        };
    }

    /// Unspecified joints retain their previous goals, as in the Python controller.
    pub fn set_targets(&mut self, targets: &Pose) -> Result<(), String> {
        let mut bounded = self.targets.clone();
        for (name, value) in targets {
            let limit = self
                .limits
                .get(name)
                .ok_or_else(|| format!("Unknown joint: {name}"))?;
            bounded.insert(name.clone(), limit.clamp(*value)?);
        }
        self.waypoints.clear();
        self.begin(Segment {
            targets: bounded,
            seconds: 0.25,
            hold: false,
        });
        self.activity = "Moving".into();
        Ok(())
    }

    pub fn reset(&mut self) -> Result<(), String> {
        self.set_targets(&self.limits.keys().map(|name| (name.clone(), 0.0)).collect())?;
        self.activity = "Resetting".into();
        Ok(())
    }

    /// Anchor a new command to observed positions; never clamp the actual physics state.
    pub fn synchronize(&mut self, positions: &Pose) -> Result<(), String> {
        let bounded = self
            .limits
            .iter()
            .map(|(name, limit)| {
                let value = positions
                    .get(name)
                    .ok_or_else(|| format!("Missing observed joint: {name}"))?;
                Ok((name.clone(), limit.clamp(*value)?))
            })
            .collect::<Result<Pose, String>>()?;
        self.positions = bounded;
        self.stop();
        Ok(())
    }

    /// Compile the entire gesture before committing, so a bad step cannot partly play.
    pub fn play(&mut self, gesture: &Gesture) -> Result<(), String> {
        let origin = self.positions.clone();
        let mut pose = origin.clone();
        let mut segments = VecDeque::new();
        for step in &gesture.steps {
            match step.kind.as_str() {
                "return_to_start" => pose = origin.clone(),
                "targets" | "offsets" => {
                    for (name, angle) in &step.angles {
                        let limit = self
                            .limits
                            .get(name)
                            .ok_or_else(|| format!("Unknown joint: {name}"))?;
                        let mut target = *angle;
                        if step.kind == "offsets" {
                            target = origin[name] + angle;
                            if step.on_limit == "reverse"
                                && !(limit.lower <= target && target <= limit.upper)
                            {
                                target = origin[name] - angle;
                            }
                        }
                        pose.insert(name.clone(), limit.clamp(target)?);
                    }
                }
                "hold_seconds" => {}
                kind => return Err(format!("Unknown gesture step: {kind}")),
            }
            if !step.seconds.is_finite() || !(0.0 < step.seconds && step.seconds <= 60.0) {
                return Err("Step duration must be finite and between 0 and 60 seconds".into());
            }
            segments.push_back(Segment {
                targets: pose.clone(),
                seconds: step.seconds,
                hold: step.kind == "hold_seconds",
            });
        }
        let first = segments
            .pop_front()
            .ok_or("A gesture needs at least one step")?;
        self.waypoints = segments;
        self.begin(first);
        self.activity = format!("Playing {}", gesture.label);
        Ok(())
    }

    pub fn stop(&mut self) {
        self.waypoints.clear();
        self.targets = self.positions.clone();
        self.start = self.positions.clone();
        self.elapsed = 0.0;
        self.duration = 0.0;
        self.activity = "Stopped".into();
    }

    pub fn step(&mut self, dt: f64) -> Result<Pose, String> {
        if !dt.is_finite() || dt < 0.0 {
            return Err("dt must be finite and nonnegative".into());
        }
        let mut dt = dt * self.speed;
        while dt > 0.0 && self.moving() {
            let used = dt.min(self.duration - self.elapsed);
            self.elapsed += used;
            dt -= used;
            let fraction = (self.elapsed / self.duration).min(1.0);
            let eased = fraction * fraction * (3.0 - 2.0 * fraction);
            self.positions = self
                .targets
                .iter()
                .map(|(name, value)| {
                    (
                        name.clone(),
                        self.start[name] + (value - self.start[name]) * eased,
                    )
                })
                .collect();
            if self.elapsed >= self.duration {
                self.positions = self.targets.clone();
                if let Some(segment) = self.waypoints.pop_front() {
                    self.begin(segment);
                } else {
                    self.activity = "Ready".into();
                }
            }
        }
        Ok(self.positions.clone())
    }
}
