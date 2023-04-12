use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

use ndarray::{Array1, Array2, arr1, arr2};
use regex::Regex;
use serde::Deserialize;
use sparse::{CsMat, CsMatOwned, SparseMat};

#[derive(Debug, Deserialize)]
struct Message {
    mess: String,
    target: usize,
}

fn read_data(file_path: &PathBuf) -> Vec<Message> {
    let file = File::open(file_path).unwrap();
    let reader = BufReader::new(file);

    reader
        .lines()
        .map(|line| {
            let line = line.unwrap();
            let fields: Vec<&str> = line.split(',').collect();
            Message {
                mess: fields[0].to_owned(),
                target: fields[1].parse().unwrap(),
            }
        })
        .collect()
}

fn text_process(mess: &str, stop_words: &HashMap<String, bool>) -> Vec<String> {
    let re = Regex::new(r"[^\w\s]").unwrap();

    let cleaned_mess: Vec<String> = re
        .replace_all(&mess, "")
        .split_whitespace()
        .map(|word| word.to_lowercase())
        .filter(|word| !stop_words.contains_key(word))
        .collect();

    cleaned_mess
}

fn tokenize(messages: &[Message]) -> (Vec<Vec<String>>, Array1<usize>) {
    let stop_words: HashMap<String, bool> = {
        let file = File::open("./stopwords.txt").unwrap();
        let reader = BufReader::new(file);

        reader
            .lines()
            .map(|line| line.unwrap())
            .map(|word| (word, true))
            .collect()
    };

    let mut tokenized_messages = Vec::new();
    let mut targets = Vec::new();

    for message in messages {
        let tokenized = text_process(&message.mess, &stop_words);
        if !tokenized.is_empty() {
            tokenized_messages.push(tokenized);
            targets.push(message.target);
        }
    }

    let max_len = tokenized_messages.iter().map(|tokens| tokens.len()).max().unwrap();

    let mut token_matrix = Array2::<usize>::zeros((tokenized_messages.len(), max_len));

    for (i, tokens) in tokenized_messages.iter().enumerate() {
        for (j, token) in tokens.iter().enumerate() {
            let word_id = bow_transformer.get(token).unwrap();
            token_matrix[[i, j]] = word_id;
        }
    }

    (tokenized_messages, arr1(&targets))
}

fn normalize(messages: &[Vec<String>]) -> CsMatOwned<f32> {
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut data = Vec::new();

    let vocab_size = bow_transformer.len();

    for (i, tokens) in messages.iter().enumerate() {
        let mut word_counts = HashMap::new();
        for token in tokens {
            let word_id = bow_transformer.get(token).unwrap();
            let count = word_counts.entry(word_id).or_insert(0);
            *count += 1;
        }
        for (word_id, count) in word_counts {
            let tf_idf = (count as f32) * idf_transformer[word_id];
            rows.push(i);
            cols.push(word_id);
            data.push(tf_idf);
        }
    }

    CsMatOwned::new((messages.len(), vocab_size), rows).unwrap()
        .insert_cols(cols, data)
        .unwrap()
}