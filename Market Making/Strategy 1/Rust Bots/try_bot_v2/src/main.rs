use std::error::Error;
mod ticks;
mod signal;
mod pricemodel;
mod hyperparameters;
use rust_decimal::Decimal;
use tokio::task;
use url::ParseError;
use crossbeam_channel::{bounded, select, Sender, Receiver};
use std::time::Duration;

async fn process_signal(rx: Receiver<[Decimal; 2]>) {
        // Use the select! macro to receive with a timeout
    loop {
        select! {
            recv(rx) -> msg => if let Ok(data) = msg { 
                println!("{:?}", data) 
            },
            default(Duration::from_millis(99)) => {},
        }
    }
}


#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let symbol = "btcusdt";
    let model_params = pricemodel::params_arr();
    let (tx1, rx1): (Sender<[Decimal; 2]>, Receiver<[Decimal; 2]>) = bounded(1);
    let tx = tx1.clone();
    let rx = rx1.clone();
    task::spawn(process_signal(rx));
    let _ = Ok::<(), ParseError>(task::spawn(ticks::open_websocket(symbol.into(), model_params, tx)).await??);
    Ok(())
}