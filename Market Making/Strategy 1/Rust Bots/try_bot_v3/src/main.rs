use std::error::Error;
mod ticks;
mod signal;
mod model;
mod placeorder;
mod hyperparameters;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use tokio::task;
use url::ParseError;
use crossbeam_channel::{bounded, select, Sender, Receiver};
use std::time::Duration;

async fn process_order(rx: Receiver<[Decimal; 3]>) {
    loop {
        let mut side = "";
        // Use the select! macro to receive with a timeout
        select! {
            recv(rx) -> msg => if let Ok(data) = msg {
                println!("{:?}", data);
                let s= [data[0].to_string(), data[1].to_string(), data[2].to_string()];
                if s[1] == "1" {
                    side = "BUY";
                }
                else {
                    side = "SELL";
                }
                // Send a signal to the processing thread
                match placeorder::create_limit_order([&s[0], side, &s[2]]) {
                    Ok(()) => {
                    },
                    Err(err) => {
                        eprintln!("Error ordering: {}", err);
                    }
                };
            },
            default(Duration::from_millis(99)) => {},
        }
    }
}

async fn process_signal(rx: Receiver<[Decimal; 3]>, tx: Sender<[Decimal; 3]>) {
    let mut dummy = dec!(0);
    loop {
        // Use the select! macro to receive with a timeout
        select! {
            recv(rx) -> msg => if let Ok(data) = msg {
                if data[1] != dummy {
                    dummy = data[1];
                    match tx.try_send(data) {
                        Ok(()) => {
                        },
                        Err(_) => println!("Sender error Alpha: Receiver may be closed or channel is full"),
                    }
                }
            },
            default(Duration::from_millis(99)) => {},
        }
    }
}


#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let symbol = hyperparameters::SYMBOL;
    // Get Model parameters
    let model_params = model::params_arr();
    // Create channel for cross thread communication
    let (tx_tick, rx_tick): (Sender<[Decimal; 3]>, Receiver<[Decimal; 3]>) = bounded(1);
    let tx = tx_tick.clone();
    let rx = rx_tick.clone();
    let (tx_alpha, rx_alpha): (Sender<[Decimal; 3]>, Receiver<[Decimal; 3]>) = bounded(1);
    let tx0 = tx_alpha.clone();
    // Spawn Signal Thread
    task::spawn(process_signal(rx, tx0));
    // Spawn Order Thread (Only 2 because I have only four threads)
    let rx0 = rx_alpha.clone();
    tokio::spawn(process_order(rx0));
    let rx1 = rx_alpha.clone();
    tokio::spawn(process_order(rx1));
    // Spawm Tick Thread
    let _ = Ok::<(), ParseError>(task::spawn(ticks::open_websocket(symbol.into(), model_params, tx)).await??);
    Ok(())
}