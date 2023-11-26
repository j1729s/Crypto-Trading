use rust_decimal::Decimal;
use rust_decimal_macros::dec;

pub const LAGS: usize = 2;
pub const DELAYS: usize = 4;
pub const SIZE: usize = 4 + 2 * LAGS;
pub const THRESHOLD: Decimal = dec!(0.1);