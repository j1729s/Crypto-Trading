use crossbeam::channel::Sender;
use futures_util::{stream::StreamExt, SinkExt};
use std::borrow::Cow;
use std::str::FromStr;
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::Message;
use url::{ParseError, Url};
use rust_decimal::Decimal;
use serde_json::{json, Value};
use serde::Deserialize;
use std::collections::BTreeMap;

#[derive(Deserialize, Debug)]
struct OrderBookStream {
    u: i128,
    a: Vec<Vec<String>>,
    b: Vec<Vec<String>>
}

#[derive(Deserialize, Debug)]
struct TradeStream {
    T: i128,
    S: String,
    p: String,
    v: String
}

fn update_state_with_message(state: &mut BTreeMap<Decimal, Decimal>, message: Vec<Vec<String>>) {
    for row in message {
        if let Ok(price) = Decimal::from_str(&row[0]) {
            if let Ok(volume) = Decimal::from_str(&row[1]) {
                // Insert or update price-volume pair
                state.insert(price, volume);

                // Remove price if volume is zero
                if volume.is_zero() {
                    state.remove(&price);
                }
            }
        }
    }
}

pub async fn handle_orderbook(symbol: Cow<'static, str>, 
market: Cow<'static, str>, 
sendb: Sender<BTreeMap<Decimal, Decimal>>, 
senda: Sender<BTreeMap<Decimal, Decimal>>) -> Result<(), ParseError> {
    // Initialize state with an empty BTreeMap
    let mut bids: BTreeMap<Decimal, Decimal> = BTreeMap::new();
    let mut asks: BTreeMap<Decimal, Decimal> = BTreeMap::new();

    let url = format!("wss://stream.bybit.com/v5/public/{market}");  
    let url = Url::parse(&url)?;
    
    let mut depth = "200";

    if &market.to_string() == "linear" {
        depth = "500";
    }

    let orderbook_stream = format!("orderbook.{depth}.{symbol}");

    let ping_request = json!({
        "req_id": "test",
        "op": "ping",
    });

    let orderbook_request = json!({
        "req_id": "test",
        "op": "subscribe",
        "args": [
            &orderbook_stream
        ]
    });

    loop {
        let mut u: i128 = 1;
        let (mut ws_stream, _) = connect_async(url.clone()).await.expect("Failed to connect");
        println!("WebSocket handshake has been successfully completed");
    
        let _ = ws_stream.send(Message::Text(orderbook_request.to_string())).await;
        let ping = Message::Ping(ping_request.to_string().into());

        let mut counter = 0;

        while let Some(msg) = ws_stream.next().await {
            match msg {
                Ok(msg) => {
                    if let Message::Text(text) = msg {
                        let data: Value = serde_json::from_str(&text).unwrap();
                        if let Some(topic) = data.get("topic") {
                            if counter > 100 {
                                counter = 0;
                                match ws_stream.send(ping.clone()).await {
                                    Ok(()) => {
                                    println!("Ping sent successfully");
                                    }
                                    Err(err) => {
                                        eprintln!("Error sending ping: {}", err);
                                    }
                                }
                            }   
                            if topic.as_str() == Some(&orderbook_stream) {
                                //println!("{}", data);
                                counter += 1;
                                let book: OrderBookStream = serde_json::from_value(data["data"].clone()).expect("OrderBook error");
                                if book.u == 1 {
                                    println!("New Snapshot")
                                }
                                if book.u > u || book.u == 1{
                                    u = book.u;
                                    update_state_with_message(&mut bids, book.b);
                                    match sendb.try_send(bids.clone()) {
                                        Ok(()) => {
                                        },
                                        Err(_) => println!("Sender error Bids: Receiver may be closed or channel is full"),
                                    }
                                    update_state_with_message(&mut asks, book.a);
                                    match senda.try_send(asks.clone()) {
                                        Ok(()) => {
                                        },
                                        Err(_) => println!("Sender error Asks: Receiver may be closed or channel is full"),
                                    }
                                }
                            } else {
                                print!("Unknown")
                            }
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Error while receiving message: {e}");
                }
            }
        }
    } 
}