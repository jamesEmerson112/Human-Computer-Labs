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
    pace_limit: f64,
    waypoints: VecDeque<Segment>,
    speed: f64,
    stream_velocity: Pose,
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
            pace_limit: f64::INFINITY,
            waypoints: VecDeque::new(),
            speed: 1.0,
            stream_velocity: BTreeMap::new(),
        })
    }

    pub fn speed(&self) -> f64 {
        self.speed
    }

    /// Requested pace up to 3x; each segment still respects its velocity envelope.
    pub fn set_speed(&mut self, speed: f64) -> Result<(), String> {
        if !speed.is_finite() || !(0.0 < speed && speed <= 3.0) {
            return Err("Playback speed must be greater than zero and at most three".into());
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
        let minimum_duration = self.limits.iter().fold(0.0_f64, |duration, (name, limit)| {
            duration.max(1.5 * (self.targets[name] - self.positions[name]).abs() / limit.velocity)
        });
        self.duration = if segment.hold {
            segment.seconds
        } else {
            0.25_f64.max(segment.seconds).max(minimum_duration)
        };
        self.pace_limit = if minimum_duration > 0.0 {
            self.duration / minimum_duration
        } else {
            f64::INFINITY
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
        self.stream_velocity.clear();
        self.waypoints.clear();
        self.targets = self.positions.clone();
        self.start = self.positions.clone();
        self.elapsed = 0.0;
        self.duration = 0.0;
        self.activity = "Stopped".into();
    }

    /// Follow a wall-clock cue without queuing historical segments. This path
    /// deliberately ignores gesture pace: synchronized speech owns its timing.
    /// Validation happens before mutation, and every physical tick limits both
    /// requested velocity and its change. Actual MuJoCo tracking remains separate.
    pub fn stream_targets(
        &mut self,
        targets: &Pose,
        dt: f64,
        velocity_scale: f64,
    ) -> Result<Pose, String> {
        if !dt.is_finite()
            || dt <= 0.0
            || !velocity_scale.is_finite()
            || !(0.0 < velocity_scale && velocity_scale <= 1.0)
        {
            return Err(
                "Streaming needs a positive finite step and velocity scale in (0, 1]".into(),
            );
        }
        if targets.len() != self.limits.len() {
            return Err("Streaming requires every joint".into());
        }
        let bounded = self
            .limits
            .iter()
            .map(|(name, limit)| {
                Ok((
                    name.clone(),
                    limit.clamp(
                        *targets
                            .get(name)
                            .ok_or_else(|| format!("Missing joint: {name}"))?,
                    )?,
                ))
            })
            .collect::<Result<Pose, String>>()?;
        for (name, limit) in &self.limits {
            let cap = limit.velocity * velocity_scale;
            let desired = ((bounded[name] - self.positions[name]) / 0.12).clamp(-cap, cap);
            let previous = self.stream_velocity.get(name).copied().unwrap_or(0.0);
            let velocity = desired.clamp(
                previous - limit.velocity * 3.0 * dt,
                previous + limit.velocity * 3.0 * dt,
            );
            let position = (self.positions[name] + velocity * dt).clamp(limit.lower, limit.upper);
            self.stream_velocity
                .insert(name.clone(), (position - self.positions[name]) / dt);
            self.positions.insert(name.clone(), position);
        }
        self.waypoints.clear();
        self.elapsed = 0.0;
        self.duration = 0.0;
        self.targets = bounded;
        self.start = self.positions.clone();
        Ok(self.positions.clone())
    }

    pub fn step(&mut self, dt: f64) -> Result<Pose, String> {
        if !dt.is_finite() || dt < 0.0 {
            return Err("dt must be finite and nonnegative".into());
        }
        // Keep ordinary/slower playback arithmetic unchanged. Above 1x each
        // segment consumes wall time at its own velocity-limited clock rate.
        let mut dt = dt * self.speed.min(1.0);
        while dt > 0.0 && self.moving() {
            let scale = if self.speed > 1.0 {
                self.speed.min(self.pace_limit)
            } else {
                1.0
            };
            let used = (dt * scale).min(self.duration - self.elapsed);
            self.elapsed += used;
            dt -= used / scale;
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
