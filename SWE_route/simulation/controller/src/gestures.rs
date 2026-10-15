//! Parse the existing inert JSON format, rejecting duplicate keys at every depth.

use serde::{
    Deserialize, Deserializer, Serialize,
    de::{self, MapAccess, SeqAccess, Visitor},
};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt, fs,
    path::Path,
};

#[derive(Clone, Debug, Serialize)]
pub struct GestureStep {
    pub kind: String,
    pub angles: Vec<(String, f64)>,
    pub seconds: f64,
    pub on_limit: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct Gesture {
    pub name: String,
    pub label: String,
    pub description: String,
    pub steps: Vec<GestureStep>,
    pub group: String,
    /// SHA-256 of the original UTF-8 JSON bytes, including whitespace and BOM.
    /// Runtime metadata only: the authored gesture schema is unchanged.
    pub revision: String,
}

struct UniqueValue(Value);

impl<'de> Deserialize<'de> for UniqueValue {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct UniqueVisitor;
        impl<'de> Visitor<'de> for UniqueVisitor {
            type Value = UniqueValue;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("JSON with unique object keys")
            }
            fn visit_bool<E: de::Error>(self, value: bool) -> Result<Self::Value, E> {
                Ok(UniqueValue(value.into()))
            }
            fn visit_i64<E: de::Error>(self, value: i64) -> Result<Self::Value, E> {
                Ok(UniqueValue(value.into()))
            }
            fn visit_u64<E: de::Error>(self, value: u64) -> Result<Self::Value, E> {
                Ok(UniqueValue(value.into()))
            }
            fn visit_f64<E: de::Error>(self, value: f64) -> Result<Self::Value, E> {
                serde_json::Number::from_f64(value)
                    .map(|v| UniqueValue(Value::Number(v)))
                    .ok_or_else(|| E::custom("JSON numbers must be finite"))
            }
            fn visit_str<E: de::Error>(self, value: &str) -> Result<Self::Value, E> {
                Ok(UniqueValue(value.into()))
            }
            fn visit_string<E: de::Error>(self, value: String) -> Result<Self::Value, E> {
                Ok(UniqueValue(value.into()))
            }
            fn visit_unit<E: de::Error>(self) -> Result<Self::Value, E> {
                Ok(UniqueValue(Value::Null))
            }
            fn visit_none<E: de::Error>(self) -> Result<Self::Value, E> {
                Ok(UniqueValue(Value::Null))
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                let mut values = Vec::new();
                while let Some(value) = seq.next_element::<UniqueValue>()? {
                    values.push(value.0);
                }
                Ok(UniqueValue(Value::Array(values)))
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
                let mut values = Map::new();
                while let Some(key) = map.next_key::<String>()? {
                    if values.contains_key(&key) {
                        return Err(de::Error::custom(format!("duplicate JSON key: {key}")));
                    }
                    values.insert(key, map.next_value::<UniqueValue>()?.0);
                }
                Ok(UniqueValue(Value::Object(values)))
            }
        }
        deserializer.deserialize_any(UniqueVisitor)
    }
}

fn object<'a>(data: &'a Value, location: &str) -> Result<&'a Map<String, Value>, String> {
    data.as_object()
        .ok_or_else(|| format!("{location}: expected an object"))
}

fn allowed_keys(data: &Map<String, Value>, keys: &[&str], location: &str) -> Result<(), String> {
    if let Some(key) = data.keys().find(|key| !keys.contains(&key.as_str())) {
        return Err(format!("{location}: unexpected property {key}"));
    }
    Ok(())
}

fn text_field(
    data: &Map<String, Value>,
    key: &str,
    required: bool,
    max: usize,
    nonblank: bool,
) -> Result<String, String> {
    let Some(value) = data.get(key) else {
        return if required {
            Err(format!("gesture: missing {key}"))
        } else {
            Ok(String::new())
        };
    };
    let value = value
        .as_str()
        .ok_or_else(|| format!("{key}: expected a string"))?;
    // Python's Unicode regex \s also includes the four information separator controls.
    if value.chars().count() > max
        || (nonblank
            && !value
                .chars()
                .any(|c| !c.is_whitespace() && !('\u{1c}'..='\u{1f}').contains(&c)))
    {
        return Err(format!("{key}: invalid length or blank text"));
    }
    Ok(value.to_owned())
}

fn duration(value: Option<&Value>) -> Result<f64, String> {
    let seconds = match value {
        None => 0.25,
        Some(value) => value.as_f64().ok_or("Step duration must be a number")?,
    };
    if !seconds.is_finite() || !(0.0 < seconds && seconds <= 60.0) {
        return Err("Step duration must be finite and between 0 and 60 seconds".into());
    }
    Ok(seconds)
}

/// The validation rules mirror gesture.schema.json; only validated data reaches motion.
pub fn parse_gesture(data: &Value, joint_names: &BTreeSet<String>) -> Result<Gesture, String> {
    let data = object(data, "gesture")?;
    allowed_keys(
        data,
        &[
            "$schema",
            "schema_version",
            "name",
            "label",
            "description",
            "group",
            "units",
            "steps",
        ],
        "gesture",
    )?;
    if let Some(schema) = data.get("$schema") {
        if !schema.is_string() {
            return Err("$schema: expected a string".into());
        }
    }
    if data.get("schema_version").and_then(Value::as_f64) != Some(1.0) {
        return Err("schema_version must be 1".into());
    }
    if data.get("units").and_then(Value::as_str) != Some("radians") {
        return Err("units must be radians".into());
    }
    let name = text_field(data, "name", true, 64, true)?;
    // JSON Schema's Python regex permits one terminal newline with the `$` anchor.
    let pattern_name = name.strip_suffix('\n').unwrap_or(&name);
    let mut letters = pattern_name.chars();
    if !letters.next().is_some_and(|c| c.is_ascii_lowercase())
        || !letters.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
    {
        return Err("name must match ^[a-z][a-z0-9_]*$".into());
    }
    let label = text_field(data, "label", true, 60, true)?;
    let description = text_field(data, "description", false, 500, false)?;
    let group = text_field(data, "group", false, 60, true)?;
    let values = data
        .get("steps")
        .and_then(Value::as_array)
        .ok_or("steps must be an array")?;
    if values.is_empty() || values.len() > 128 {
        return Err("steps must contain between 1 and 128 items".into());
    }
    let mut steps = Vec::new();
    for (index, value) in values.iter().enumerate() {
        let location = format!("steps.{index}");
        let data = object(value, &location)?;
        let kinds: Vec<_> = ["targets", "offsets", "hold_seconds", "return_to_start"]
            .into_iter()
            .filter(|key| data.contains_key(*key))
            .collect();
        if kinds.len() != 1 {
            return Err(format!("{location}: expected exactly one step kind"));
        }
        let kind = kinds[0];
        allowed_keys(
            data,
            match kind {
                "targets" => &["targets", "duration_seconds"],
                "offsets" => &["offsets", "duration_seconds", "on_limit"],
                "hold_seconds" => &["hold_seconds"],
                _ => &["return_to_start", "duration_seconds"],
            },
            &location,
        )?;
        let mut angles = Vec::new();
        if kind == "targets" || kind == "offsets" {
            let values = object(&data[kind], &location)?;
            if values.is_empty() {
                return Err(format!("{location}: angles must not be empty"));
            }
            for (name, value) in values {
                if !joint_names.contains(name) {
                    return Err(format!("{location}: unknown joint {name}"));
                }
                let angle = value
                    .as_f64()
                    .filter(|v| v.is_finite())
                    .ok_or_else(|| format!("{location}: angles must be finite numbers"))?;
                angles.push((name.clone(), angle));
            }
        } else if kind == "return_to_start" && data[kind].as_bool() != Some(true) {
            return Err(format!("{location}: return_to_start must be true"));
        }
        let seconds = duration(data.get(if kind == "hold_seconds" {
            "hold_seconds"
        } else {
            "duration_seconds"
        }))?;
        let on_limit = match data.get("on_limit") {
            None => "clamp",
            Some(value) => value
                .as_str()
                .filter(|v| *v == "clamp" || *v == "reverse")
                .ok_or("on_limit must be clamp or reverse")?,
        };
        steps.push(GestureStep {
            kind: kind.into(),
            angles,
            seconds,
            on_limit: on_limit.into(),
        });
    }
    Ok(Gesture {
        name,
        label,
        description,
        steps,
        group,
        revision: String::new(),
    })
}

pub(crate) fn parse_unique_json(text: &str) -> Result<Value, String> {
    serde_json::from_str::<UniqueValue>(text)
        .map(|data| data.0)
        .map_err(|e| e.to_string())
}

pub fn parse_gesture_json(text: &str, joint_names: &BTreeSet<String>) -> Result<Gesture, String> {
    let data = parse_unique_json(text.strip_prefix('\u{feff}').unwrap_or(text))?;
    let mut gesture = parse_gesture(&data, joint_names)?;
    gesture.revision = format!("{:x}", Sha256::digest(text.as_bytes()));
    Ok(gesture)
}

pub fn load_gesture(path: &Path, joint_names: &BTreeSet<String>) -> Result<Gesture, String> {
    fs::read_to_string(path)
        .map_err(|e| e.to_string())
        .and_then(|text| parse_gesture_json(&text, joint_names))
        .map_err(|e| {
            format!(
                "{}: {e}",
                path.file_name().unwrap_or_default().to_string_lossy()
            )
        })
}

/// Return a complete fresh library; callers replace the old library only on success.
pub fn load_gestures(
    joint_names: &BTreeSet<String>,
    directory: &Path,
) -> Result<BTreeMap<String, Gesture>, String> {
    let mut paths = fs::read_dir(directory)
        .map_err(|e| format!("{}: {e}", directory.display()))?
        .map(|entry| entry.map(|e| e.path()))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())?;
    paths.retain(|path| {
        path.extension().is_some_and(|ext| {
            if cfg!(windows) {
                ext.to_string_lossy().eq_ignore_ascii_case("json")
            } else {
                ext == "json"
            }
        })
    });
    paths.sort();
    let mut gestures = BTreeMap::new();
    let mut labels = BTreeSet::new();
    for path in paths {
        let gesture = load_gesture(&path, joint_names)?;
        if gestures.contains_key(&gesture.name) {
            return Err(format!(
                "{}: duplicate gesture name {}",
                path.display(),
                gesture.name
            ));
        }
        if !labels.insert(gesture.label.clone()) {
            return Err(format!(
                "{}: duplicate gesture label {}",
                path.display(),
                gesture.label
            ));
        }
        gestures.insert(gesture.name.clone(), gesture);
    }
    if gestures.is_empty() {
        return Err(format!(
            "No gesture JSON files found in {}",
            directory.display()
        ));
    }
    Ok(gestures)
}
