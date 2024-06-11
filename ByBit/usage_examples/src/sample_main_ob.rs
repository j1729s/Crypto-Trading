extern crate Bybit_Helper;
use Bybit_Helper::handle_orderbook;
use rust_decimal::Decimal;
use tokio::task;
use std::collections::BTreeMap;
use std::error::Error;
use std::borrow::Cow;
use crossbeam::channel::{bounded, Receiver, Sender, select};
use url::ParseError;
use std::time::Duration;

const WAIT: u64 = 99;

async fn process_bids(recb: Receiver<BTreeMap<Decimal, Decimal>>) {
    loop {
        select! {
            recv(recb) -> msg => {
                if let Ok(msg) = msg {
                    println!("Bids received: {:?}", msg);
                } else {
                    println!("Error receiving bids");
                }
            },
            default(Duration::from_millis(WAIT)) => {},
        }
    }
}

async fn process_asks(reca: Receiver<BTreeMap<Decimal, Decimal>>) {
    loop {
        select! {
            recv(reca) -> msg => {
                if let Ok(msg) = msg {
                    println!("Asks received: {:?}", msg);
                } else {
                    println!("Error receiving asks");
                }
            },
            default(Duration::from_millis(WAIT)) => {},
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>>{
    // Create two channels for communication
    let (sender_b, receiver_b): (Sender<BTreeMap<Decimal, Decimal>>, Receiver<BTreeMap<Decimal, Decimal>>) = bounded(1);
    let (sender_a, receiver_a): (Sender<BTreeMap<Decimal, Decimal>>, Receiver<BTreeMap<Decimal, Decimal>>) = bounded(1);
    let send_b = sender_b.clone();
    let send_a = sender_a.clone();
    //let recv_b = receiver_b.clone();
    //let recv_a = receiver_a.clone();
    task::spawn(process_bids(receiver_b.clone()));
    task::spawn(process_asks(receiver_a.clone()));
    let _ = Ok::<(), ParseError>(task::spawn(handle_orderbook(Cow::Borrowed("BTCUSDT"), Cow::Borrowed("linear"), send_b, send_a)).await??);
    Ok(())
}