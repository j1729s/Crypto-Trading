use futures_util::stream::StreamExt;
use rust_decimal_macros::dec;
use std::borrow::Cow;
use std::time::Duration;
use tokio::time::sleep;
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::Message;
use url::{ParseError, Url};
use fxhash::FxHashMap;
use rust_decimal::Decimal;
use serde::Deserialize;
use serde_with::{serde_as, DisplayFromStr};
use crossbeam_channel::Sender;

use crate::hyperparameters;
use crate::signal;

const LAGS: usize = hyperparameters::LAGS;
const SIZE: usize = hyperparameters::SIZE;
const RECONNECT_DELAY: u64 = 5;

#[serde_as]
#[derive(Debug, Deserialize)]
struct BookTickerData {
    #[serde_as(as = "DisplayFromStr")]
    b: Decimal,
    #[serde_as(as = "DisplayFromStr")]
    B: Decimal,
    #[serde_as(as = "DisplayFromStr")]
    a: Decimal,
    #[serde_as(as = "DisplayFromStr")]
    A: Decimal,
}

#[serde_as]
#[derive(Debug, Deserialize)]
struct AggTradeData {
    #[serde_as(as = "DisplayFromStr")]
    p: Decimal,
    #[serde_as(as = "DisplayFromStr")]
    q: Decimal,
}

async fn process_stream_data(input: &str, state: &mut FxHashMap<String, Decimal>) -> Result<[Decimal; 7], ()> {
    let mut result_array = [Default::default(); 7];
    match serde_json::from_str::<serde_json::Value>(input) {
        Ok(parsed) => {
            let stream_type = parsed["stream"].as_str().unwrap_or("");
            let data = &parsed["data"];
            // Round "E" up to the nearest multiple of 100.
            if let Some(new_e) = data["E"].as_f64() {
                let rounded_e = Decimal::from_f64_retain((new_e / 100.0).ceil() * 100.0).unwrap();
                // Compare the new "E" value with the one stored in the state.
                if let Some(&old_e) = state.get("E") {
                    if rounded_e > old_e && state.len() == 7 {
                        // Convert the HashMap into an array with the desired order.
                        result_array = convert_to_array(&state);
                        state.insert("E".to_string(), rounded_e);
                        if state.len() == 7 {
                            // Print the updated values in the array.
                            return Ok(result_array);
                        }
                    }
                } else {
                    // If "E" is not in the state, insert it.
                    state.insert("E".to_string(), rounded_e);
                }
            }
            match stream_type {
                "btcusdt@bookTicker" => {
                    let book_data: BookTickerData = serde_json::from_value(data.clone()).expect("Book Ticker error");
                    state.insert("b".to_string(), book_data.b);
                    state.insert("B".to_string(), book_data.B);
                    state.insert("a".to_string(), book_data.a);
                    state.insert("A".to_string(), book_data.A);
                }
                "btcusdt@aggTrade" => {
                    let agg_trade_data: AggTradeData = serde_json::from_value(data.clone()).expect("Agg Trade Error");
                    state.insert("p".to_string(), agg_trade_data.p);
                    if let Some(&old_q) = state.get("q") {
                        // Add the new "q" value to the old "q" value.
                        let new_q = old_q + agg_trade_data.q;
                        // Insert the rounded  number to prevent floating point errors
                        state.insert("q".to_string(), new_q);
                    } else {
                        // If "q" is not in the state, insert it.
                        state.insert("q".to_string(), agg_trade_data.q);
                    }
                }
                _ => {
                    // Handle other stream types or unknown stream types
                }
            }
        },
        Err(e) => {
            eprintln!("Error parsing JSON: {}", e);
        },
    }
    Err(())
}

fn convert_to_array(state: &FxHashMap<String, Decimal>) -> [Decimal; 7] {
    let mut result: [Decimal; 7] = [Decimal::default(); 7];
    let fields = ["E", "b", "B", "a", "A", "p", "q"];

    for (i, field) in fields.iter().enumerate() {
        if let Some(value) = state.get(*field) {
            result[i] = *value;
        }
    }
    result
}

fn slice_to_array(slice: &[Decimal]) -> [Decimal; 6] {
    slice.try_into().expect("slice with incorrect length")
}

pub async fn open_websocket(symbol: Cow<'static, str>, model_params: [Decimal; SIZE], tx: Sender<[Decimal; 2]>) -> Result<(), ParseError> {
    let mut row = [Decimal::default(); 6];
    let mut calculator = signal::MetricsCalculator::new(LAGS, model_params);
    // Create an initial state with "E" as 0.
    let mut state = FxHashMap::default();
    let url =
        format!("wss://fstream.binance.com/stream?streams={symbol}@bookTicker/{symbol}@aggTrade",);

    let url = Url::parse(&url)?;
    
    loop {
        match connect_async(&url).await {
            Ok((mut ws_stream, _)) => {
                while let Some(msg) = ws_stream.next().await {
                    match msg {
                        Ok(response) => {
                            if let Message::Text(text) = response {
                                // Convert the WebSocket message to a string and pass it to process_stream_data.
                                let result = process_stream_data(&text, &mut state).await;
                                if let Ok(s) = result {
                                    row = slice_to_array(&s[1..]);
                                    let signal = signal::MetricsCalculator::gen_signal(&mut calculator, row);
                                    match signal{
                                        1 => {
                                             // Send a signal to the processing thread
                                            match tx.try_send([dec!(1), s[1]]) {
                                                Ok(()) => {
                                                },
                                                Err(_) => println!("Sender error: Receiver may be closed or channel is full"),
                                            }
                                        },
                                        -1 => {
                                             // Send a signal to the processing thread
                                            match tx.try_send([dec!(-1), s[3]]) {
                                                Ok(()) => {
                                                },
                                                Err(_) => println!("Sender error: Receiver may be closed or channel is full"),
                                            }
                                        },
                                        _ => {
                                        },
                                   }
                                }
                            } else {
                                eprintln!("Received a non-text WebSocket message.");
                            }
                        }
                        Err(e) => {
                            eprintln!("Error while receiving message: {e}");
                            break;
                        }
                    }
                }
            }
            Err(e) => {
                eprintln!("WebSocket encountered an {e}");
                eprintln!("Reconnecting in {RECONNECT_DELAY} seconds...");
                sleep(Duration::from_secs(RECONNECT_DELAY)).await;
            }
        };
    }
}