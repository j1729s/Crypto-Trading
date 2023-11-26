use std::cmp::Ordering;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use crate::hyperparameters;

const LAGS: usize = hyperparameters::LAGS;
const SIZE: usize = hyperparameters::SIZE;
const THRESHOLD: Decimal = hyperparameters::THRESHOLD;


pub struct MetricsCalculator {
    pub buffer: [[Decimal; 6]; 2],
    pub atp: Decimal,
    pub metric_arr: [Decimal; 5],
    pub lag_buffer: Vec<[Decimal; 2]>,
    pub count: usize,
    pub new_row: [Decimal; 4],
    pub params: [Decimal; SIZE],
    pub mpc: Decimal
}

impl MetricsCalculator {
    pub fn new(lags: usize, params: [Decimal; SIZE]) -> MetricsCalculator {
        let lag_buffer = vec![[Decimal::default(); 2]; lags];
        let count = 1;
        MetricsCalculator {
            buffer: [[Decimal::default(); 6], [Decimal::default(); 6]],
            atp: Decimal::default(),
            metric_arr: [Decimal::default(); 5],
            lag_buffer,
            count,
            new_row: [Decimal::default(); 4],
            params,
            mpc: Decimal::default()
        }
    }

    fn slice_to_array2(slice: &[Decimal]) -> [Decimal; 2] {
        slice.try_into().expect("slice with incorrect length")
    }
    fn slice_to_array4(slice: &[Decimal]) -> [Decimal; 4] {
        slice.try_into().expect("slice with incorrect length")
    }
    
    fn np_rollup(&self) -> Vec<[Decimal; 2]> {
        // Define the number of positions to roll (negative to roll upward)
        let num_position: usize = 1;

        // Create a temporary buffer to hold the rolled values
        let mut temp_buffer = Vec::new();
        temp_buffer.extend_from_slice(&self.lag_buffer[num_position..]);
        temp_buffer.push([Decimal::default(); 2]);
        
        temp_buffer
    }

    fn cal_metrics(&self) -> [Decimal; 5] {
        let spread = self.buffer[1][2] - self.buffer[1][0];
        let diff = [
            self.buffer[1][0] - self.buffer[0][0],
            self.buffer[1][1] - self.buffer[0][1],
            self.buffer[1][2] - self.buffer[0][2],
            self.buffer[1][3] - self.buffer[0][3],
            self.buffer[1][4] - self.buffer[0][4],
            self.buffer[1][5] - self.buffer[0][5],
            ];
        let zero = Decimal::from_str_exact("0").unwrap();
        let d_bid = match diff[0].cmp(&zero) {
            Ordering::Equal => diff[1],
            Ordering::Greater => self.buffer[1][1],
            _ => zero,
        };
        let d_ask = match diff[2].cmp(&zero) {
            Ordering::Equal => diff[3],
            Ordering::Less => self.buffer[1][3],
            _ => zero,
        };
        let voi = d_bid - d_ask;
        let oir = (self.buffer[1][1] - self.buffer[1][3]) / (self.buffer[1][1] + self.buffer[1][3]);
        let tp = if diff[5] != zero {
            if diff[4] != zero {
                self.buffer[1][4] + diff[5] / diff[4]
            } else {
                self.buffer[1][4]
            }
        } else {
            self.atp
        };
        let m0 = (self.buffer[0][0] + self.buffer[0][2]) / Decimal::from_str_exact("2").unwrap();
        let m1 = (self.buffer[1][0] + self.buffer[1][2]) / Decimal::from_str_exact("2").unwrap();
        let mpb = tp - (m0 + m1) / Decimal::from_str_exact("2").unwrap();
        [tp, spread, mpb, oir, voi]
    }
    
    fn calculate_mpc(&self) -> Decimal {
        // Calculate Spread 
        let spread = self.new_row[0];
    
        // Iterate over columns in lag_buffer and flatten them into lag_row
        let mut lag_row: Vec<Decimal> = Vec::with_capacity(self.lag_buffer.len() * 2);
        for col in 0..2 {
            for row in self.lag_buffer.iter().rev() {
                lag_row.push(row[col]);
            }
        }
    
        // Create pred_row by appending new_row to [mpb, oir, voi] and lag_row
        let mut pred_row: Vec<Decimal> = Vec::with_capacity(3 + lag_row.len());
        pred_row.push(self.new_row[1]);
        pred_row.push(self.new_row[2]);
        pred_row.push(self.new_row[3]);
        pred_row.extend_from_slice(&lag_row);
        
        // Calculate mpc
        let mpc = self.params[1..].iter().zip(pred_row.iter()).fold(self.params[0], |acc, (&param, &pred_value)| {
            acc + param * pred_value / spread
        });
    
        mpc
    }

    fn print_signal(&self) -> i32 {
        if self.mpc > THRESHOLD {
            1
        } else if self.mpc < dec!(-1) * THRESHOLD {
            -1
        } else{
            0
        }
    }

    pub fn gen_signal(&mut self, row: [Decimal; 6]) -> i32 {
        if MetricsCalculator::has_zeros(&self.buffer[0]) {
            self.buffer[0] = row;
            self.atp = (self.buffer[0][0] + self.buffer[0][2]) / Decimal::from_str_exact("2").unwrap();
        } else if MetricsCalculator::has_zeros(&self.buffer[1]) {
            self.buffer[1] = row;
            self.metric_arr = self.cal_metrics();
            self.new_row = MetricsCalculator::slice_to_array4(&self.metric_arr[1..]);
            self.atp = self.metric_arr[0];
            if !self.lag_buffer.is_empty() {
                self.lag_buffer[0] = MetricsCalculator::slice_to_array2(&self.metric_arr[3..]);
            }
        } else if LAGS == 0 {
            self.buffer[0] = self.buffer[1];
            self.buffer[1] = row;
            self.metric_arr = self.cal_metrics();
            self.new_row = MetricsCalculator::slice_to_array4(&self.metric_arr[1..]);
            self.atp = self.metric_arr[0];
            self.mpc = self.calculate_mpc();
            return self.print_signal()
            
        } else if LAGS > 0 && self.count < LAGS {
            self.buffer[0] = self.buffer[1];
            self.buffer[1] = row;
            self.metric_arr = self.cal_metrics();
            self.atp = self.metric_arr[0];
            self.lag_buffer[self.count] = MetricsCalculator::slice_to_array2(&self.metric_arr[3..]);
            self.count += 1;
        } else {
            self.buffer[0] = self.buffer[1];
            self.buffer[1] = row;
            self.metric_arr = self.cal_metrics();
            self.atp = self.metric_arr[0];
            self.new_row = MetricsCalculator::slice_to_array4(&self.metric_arr[1..]);
            self.mpc = self.calculate_mpc();
            self.lag_buffer = self.np_rollup();
            self.lag_buffer[LAGS - 1] = MetricsCalculator::slice_to_array2(&self.metric_arr[3..]);
            return self.print_signal();
        }
        0
    }

    fn has_zeros<T>(row: &[T]) -> bool
    where
        T: PartialEq + Default,
    {
        for value in row {
            if *value == Default::default() {
                return true;
            }
        }
        false
    }
}