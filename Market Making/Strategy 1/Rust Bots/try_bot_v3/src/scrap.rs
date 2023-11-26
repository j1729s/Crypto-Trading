use std::error::Error;
mod ticks;
mod signal;
mod model;
mod hyperparameters;
use rust_decimal::Decimal;
use tokio::task;
use url::ParseError;
use crossbeam_channel::{bounded, select, Sender, Receiver};
use std::time::Duration;

async fn process_signal(rx: Receiver<[Decimal; 3]>) {
        // Use the select! macro to receive with a timeout
    loop {
        select! {
            recv(rx) -> msg => if let Ok(data) = msg {
                // Send a signal to the processing thread
                println!("{:?}", data)
            },
            default(Duration::from_millis(99)) => {},
        }
    }
}


#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let symbol = hyperparameters::SYMBOL;
    let model_params = model::params_arr();
    let (tx1, rx1): (Sender<[Decimal; 3]>, Receiver<[Decimal; 3]>) = bounded(1);
    let tx = tx1.clone();
    let rx = rx1.clone();
    task::spawn(process_signal(rx));
    let _ = Ok::<(), ParseError>(task::spawn(ticks::open_websocket(symbol.into(), model_params, tx)).await??);
    Ok(())
}


// Send a signal to the processing thread
let s= [data[0].to_string(), data[1].to_string(), data[2].to_string()];
match placeorder::create_limit_order([&s[0], &s[1], &s[2]]) {
    Ok(()) => {
    },
    Err(err) => {
        eprintln!("Error ordering: {}", err);
    }
};

// Spawn worker tasks
for _ in 0..3 {
    let rx0 = rx2.clone();
    tokio::spawn(async move {
        process_order(rx0).await;
    });
}
