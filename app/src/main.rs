use lambda_runtime::{run, service_fn, Error, LambdaEvent};
use rusoto_core::Region;
use rusoto_s3::{S3Client, S3};


use serde::{Deserialize, Serialize};
// import functions from lib.rs
use crate::lib::{read_data, text_process, tokenize, train_model, predict, get_accuracy, get_confusion_matrix, get_classification_report};

// define a static variable to hold the model
static mut MODEL: Option<NaiveBayes> = None;

#[derive(Deserialize)]
struct Request {
    // this is the content of a news post
    command: String,
}

#[derive(Serialize)]
struct Response {
    req_id: String,
    tc_result: String,
}

async fn function_handler(event: LambdaEvent<Request>) -> Result<Response, Error> {
    // Extract document name from the event
    let command = event.payload.command;

    // read the data from s3 bucket, file name is command
    let client = S3Client::new(Region::default());
    let bucket_name = "20newsgroups".to_string();
    let get_object_request = rusoto_s3::GetObjectRequest {
        bucket: bucket_name,
        key: command,
        ..Default::default()
    };

    // use MODEL to do text classification prediction for this news post
    let object = client.get_object(get_object_request).await.unwrap();
    let result = predict(&tokenize(&read_data(&PathBuf::from("./20news-bydate-test"))), &MODEL);

    // Prepare the response
    let resp = Response {
        req_id: event.context.request_id,
        tc_result: format!("The classification of this post is {}.", result),
    };

    // Return `Response` (it will be serialized to JSON automatically by the runtime)
    Ok(resp)
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    let client = S3Client::new(Region::default());
    let bucket_name = "20newsgroups".to_string();
    let get_object_request = rusoto_s3::GetObjectRequest {
        bucket: bucket_name,
        ..Default::default()
    };
    let object = client.get_object(get_object_request).await.unwrap();

    //analyze the data from s3 bucket, do text classification
    MODEL = train_model(&tokenize(&read_data(&PathBuf::from("./20news-bydate-train"))));

    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        // disable printing the name of the module in every log line.
        .with_target(false)
        // disabling time is handy because CloudWatch will add the ingestion time.
        .without_time()
        .init();

    run(service_fn(function_handler)).await
}
