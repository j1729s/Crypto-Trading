extern crate Bybit_Helper;
use Bybit_Helper::{handle_vol, BUFFER_SIZE};
use rust_decimal::Decimal;
use tokio::task;
use circular_buffer::CircularBuffer;
use std::error::Error;
use std::borrow::Cow;
use crossbeam::channel::{bounded, Receiver, Sender, select};
use url::ParseError;
use std::time::Duration;

const WAIT: u64 = 99;

async fn process_buffer(rect: Receiver<CircularBuffer::<BUFFER_SIZE, [Decimal; 2]>>) {
    loop {
        select! {
            recv(rect) -> msg => {
                if let Ok(msg) = msg {
                    println!("Volume Buffer is: {:?}", msg);
                } else {
                    println!("Error receiving trades buffer");
                }
            },
            default(Duration::from_millis(WAIT)) => {},
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>>{
    let (sender_t, receiver_t): (Sender<CircularBuffer::<BUFFER_SIZE, [Decimal; 2]>>, Receiver<CircularBuffer::<BUFFER_SIZE, [Decimal; 2]>>) = bounded(1);
    let send_t = sender_t.clone();
    task::spawn(process_buffer(receiver_t.clone()));
    let _ = Ok::<(), ParseError>(task::spawn(handle_vol(Cow::Borrowed("BTCUSDT"), Cow::Borrowed("linear"), send_t)).await??);
    Ok(())
}