//! Object and array structure checks for registration-time schemas.

mod additional;
mod items;
mod properties;
mod required;

pub(super) use additional::validate_additional_properties;
pub(super) use items::validate_items;
pub(super) use properties::{properties, validate_properties};
pub(super) use required::validate_required;
