use crate::Params::BUFFER_SIZE;
use crossbeam::channel::Sender;
use circular_buffer::CircularBuffer;
use futures_util::{stream::StreamExt, SinkExt};
use std::borrow::Cow;
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::Message;
use url::{ParseError, Url};
use rust_decimal::Decimal;
use serde_json::{json, Value};
use serde::Deserialize;
use rust_decimal_macros::dec;

#[derive(Debug, Deserialize)]
struct TradeStream {
    data: Vec<Trade>,
    topic: String,
    ts: i128,
}

#[derive(Debug, Deserialize)]
struct Trade {
    S: String,
    v: String,
}

fn cal_vol(buy_vol: &mut Decimal, sell_vol: &mut Decimal, trade_stream: TradeStream) {
    for trade in &trade_stream.data {
        if trade.S == "Buy" {
            *buy_vol += Decimal::from_str_exact(&trade.v).expect("Error buy_vol decimal conversion")
        } else if trade.S == "Sell" {
            *sell_vol += Decimal::from_str_exact(&trade.v).expect("Error sell_vol decimal conversion")
        }
    }
}

pub async fn handle_vol(symbol: Cow<'static, str>, 
market: Cow<'static, str>,
sendt: Sender<CircularBuffer::<BUFFER_SIZE, [Decimal; 2]>>) -> Result<(), ParseError> {

    let url = format!("wss://stream.bybit.com/v5/public/{market}");
        
    let url = Url::parse(&url)?;

    let trade_stream = format!("publicTrade.{symbol}");

    let ping_request = json!({
        "req_id": "test",
        "op": "ping",
    });

    let trades_request = json!({
        "req_id": "test",
        "op": "subscribe",
        "args": [
            &trade_stream,
        ]
    });
    
    let mut buffer = CircularBuffer::<BUFFER_SIZE, [Decimal; 2]>::new();

    loop {
        let (mut ws_stream, _) = connect_async(url.clone()).await.expect("Failed to connect");
        println!("WebSocket handshake has been successfully completed");
    
        let _ = ws_stream.send(Message::Text(trades_request.to_string())).await;
        let ping = Message::Ping(ping_request.to_string().into());

        let mut counter = 0;
        let mut rounded_T = dec!(0);
        let mut buy_vol = dec!(0);
        let mut sell_vol = dec!(0);

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
                            if topic.as_str() == Some(&trade_stream) {
                                //println!("{}", data);
                                counter += 1;
                                let trade_stream: TradeStream = serde_json::from_value(data.clone()).expect("Trades error");
                                if rounded_T == dec!(0) {
                                    rounded_T = Decimal::from_i128_with_scale(trade_stream.ts, 3).ceil() * dec!(1000);
                                    cal_vol(&mut buy_vol, &mut sell_vol, trade_stream);
                                } else if Decimal::from_i128_with_scale(trade_stream.ts, 0) <= rounded_T {
                                    cal_vol(&mut buy_vol, &mut sell_vol, trade_stream);
                                } else if Decimal::from_i128_with_scale(trade_stream.ts, 0) > rounded_T {
                                    buffer.push_back([buy_vol, sell_vol]);
                                    //println!("Buffer is: {:?}", buffer);
                                    buy_vol = dec!(0);
                                    sell_vol = dec!(0);
                                    rounded_T = Decimal::from_i128_with_scale(trade_stream.ts, 3).ceil() * dec!(1000);
                                    cal_vol(&mut buy_vol, &mut sell_vol, trade_stream);
                                    match sendt.try_send(buffer.clone()) {
                                        Ok(()) => {
                                        },
                                        Err(_) => println!("Sender error Buffer: Receiver may be closed or channel is full"),
                                    }
                                }
                            } else {
                                print!("Unknown Stream")
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

