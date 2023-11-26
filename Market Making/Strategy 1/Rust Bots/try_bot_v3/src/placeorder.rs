use hmac::{Hmac, Mac};
use sha2::Sha256;
use std::collections::BTreeMap;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::hyperparameters;


pub fn create_limit_order(order: [&str; 3]) -> Result<(), Box<dyn std::error::Error>> {
    // Use a .env file to hide this
    let api_key = "leboXxbCgKVkCLPGwxBNWQCwHEiXLo0vNOZaQloJ0gDZ5UUuvelyH7HU9thgxxEE";
    let api_secret = "cIx4Kx01LwHMYToUv62ILDjQIri0s5Sz44FXClDeQcv1Aiygg9mQMC4KnKZOTFxH";

    let mut params = BTreeMap::new();

    let symbol = hyperparameters::SYMBOL.to_uppercase();
    params.insert("symbol", symbol.as_str());

    params.insert("side", order[1]);

    params.insert("type", "LIMIT");

    let position = hyperparameters::POSITION.to_string();
    params.insert("quantity", position.as_str());

    params.insert("price", order[2]);

    params.insert("timeInForce", "GTX");


    params.insert("timestamp", order[0]);

    let signature = generate_signature(api_secret, &params);
    params.insert("signature", &signature);
    let order_json = tree_to_string(&params);

    let url = "https://fapi.binance.com/fapi/v1/order";

    let response = ureq::post(url)
        .set("Content-Type", "application/json")
        .set("X-MBX-APIKEY", api_key)
        .set("X-MBX-SIGNATURE", &signature)  // Add the signature to the request headers
        .send_string(&order_json)?;
    
    // Get the response code
    let response_code = response.status();

    if response_code != 200 {
        println!("{:?}", response.into_string()?);
    }

    Ok(())
}

fn tree_to_string(params: &BTreeMap<&str, &str>) -> String {
    let message = params.iter()
        .map(|(key, value)| format!("{}={}", key, value))
        .collect::<Vec<String>>()
        .join("&");
    message
}

fn generate_signature(secret: &str, params: &BTreeMap<&str, &str>) -> String {
    let message = tree_to_string(params);

    let mut mac = Hmac::<Sha256>::new_from_slice(secret.as_bytes()).unwrap();
    mac.update(message.as_bytes());

    let result = mac.finalize();
    let code = result.into_bytes();

    hex::encode(code)
}
fn timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards")
        .as_millis() as u64
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let timestamp = timestamp().to_string();
    create_limit_order([&timestamp, "BUY", "26000"])
}