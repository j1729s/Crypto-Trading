use rust_decimal::Decimal;
use rust_decimal_macros::dec;

pub const SYMBOL: &str = "btcusdt";
pub const POSITION: &str = "0.001";
pub const LAGS: usize = 1;
pub const DELAYS: usize = 2;
pub const SIZE: usize = 4 + 2 * LAGS;
pub const THRESHOLD: Decimal = dec!(0.1);