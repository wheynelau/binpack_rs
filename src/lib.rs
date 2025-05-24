use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};
use std::collections::HashMap;

pub mod common;
pub mod packing;
pub mod strategy;
use common::{Histogram, IFileHandles, LossMask, Sequence};

use strategy::common::fill_packing_strategy;
use strategy::iterator::PyReturnIter;
use strategy::nemo::NemoOptions;

#[derive(IntoPyObject)]
pub enum ReturnFormat {
    Composer(HashMap<String, Vec<Sequence>>),
    // Nemo has the same format, but the keys are different
    // Different entries
    Nemo(HashMap<String, NemoFormat>),
    Iterator(PyReturnIter),
}

#[derive(IntoPyObject)]
pub enum NemoFormat {
    LossMask(Vec<LossMask>),
    Tokens(Vec<Sequence>),
}

impl std::str::FromStr for ReturnFormat {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "nemo" => Ok(ReturnFormat::Nemo(HashMap::new())),
            "composer" => Ok(ReturnFormat::Composer(HashMap::new())),
            "iterator" => Ok(ReturnFormat::Iterator(PyReturnIter {
                iter: Vec::new().into_iter(),
            })),
            _ => Err("Invalid return format"),
        }
    }
}
// TODO: Consider using the seq_lens from datasets
#[allow(dead_code)]
#[derive(FromPyObject)]
enum InputFormat {
    InputIds(Vec<Sequence>),
    SeqLen(Sequence),
}
/// Formats the sum of two numbers as string.
#[pyfunction]
#[pyo3(signature = (examples, target_pack_size, packing_algorithm, return_format, pad_id, **kwargs))]
fn fast_pack(
    examples: HashMap<String, InputFormat>,
    target_pack_size: usize,
    packing_algorithm: String,
    return_format: String,
    pad_id: Option<u32>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<ReturnFormat> {
    let (sequences, seq_lens) = create_hist(&examples, target_pack_size);
    let packing_algorithm = match packing_algorithm
        .parse::<packing::PackingAlgo>() {
        Ok(packing_algorithm) => packing_algorithm,
        Err(_) => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Invalid packing algorithm. Use 'first_fit', 'first_fit_shuffle', or 'first_fit_decreasing'.",
            ))
        }
    };

    let assignments = create_packing_strategy(seq_lens, target_pack_size, packing_algorithm);
    let result = match return_format.as_str() {
        "composer" => {
            // Composer does not need answer_start_id, etc.
            fill_packing_strategy(
                assignments,
                sequences,
                target_pack_size,
                pad_id,
                ReturnFormat::Composer(HashMap::new()),
                None,
            )
        }
        "nemo" => {
            // Extract Nemo-specific kwargs from kwargs dict
            let options = NemoOptions::builder().from_py_dict(kwargs)?.build()?;

            fill_packing_strategy(
                assignments,
                sequences,
                target_pack_size,
                pad_id,
                ReturnFormat::Nemo(HashMap::new()),
                Some(options),
            )
        }
        "iterator" => {
            // Extract Nemo-specific kwargs from kwargs dict
            let options = NemoOptions::builder().from_py_dict(kwargs)?.build()?;

            fill_packing_strategy(
                assignments,
                sequences,
                target_pack_size,
                pad_id,
                ReturnFormat::Iterator(PyReturnIter {
                    iter: Vec::new().into_iter(),
                }),
                Some(options),
            )
        }
        _ => return Err(PyValueError::new_err("Unknown format")),
    };

    Ok(result)
}

fn create_hist(
    dataset: &HashMap<String, InputFormat>,
    truncate_seq_len: usize,
) -> (Histogram, Vec<usize>) {
    let mut sequences: HashMap<usize, Vec<HashMap<String, Sequence>>> = HashMap::new();
    let mut counts = vec![0u32; truncate_seq_len + 1];
    let mut seq_lens: Vec<usize> = Vec::new();

    // format the input data into a list of dicts
    let dataset = dataset
        .iter()
        .flat_map(|(key, value)| match value {
            InputFormat::InputIds(vec_seq) => vec_seq
                .iter()
                .map(move |v| {
                    let mut entry = HashMap::new();
                    entry.insert(key.clone(), v.clone());
                    entry
                })
                .collect::<Vec<_>>(),
            _ => Vec::new(),
        })
        .collect::<Vec<_>>();

    dataset.into_iter().for_each(|entry| {
        // Only need input_ids key
        let seq = entry
            .get("input_ids")
            .expect("Expected key 'input_ids' in the dataset entry");
        let seq_len = seq.len();
        // Should we check if the inputs were truncated?
        if seq_len > truncate_seq_len {
            panic!("Sequence length exceeds the maximum allowed length.");
        }
        sequences.entry(seq_len).or_default().push(entry);
        counts[seq_len] += 1;
    });

    for seq_len in 0..(truncate_seq_len + 1) {
        let seq_len = sequences.get(&seq_len).map_or(0, |v| v.len());
        seq_lens.push(seq_len);
    }

    (sequences, seq_lens)
}

fn create_packing_strategy(
    histogram: Vec<usize>,
    pack_size: usize,
    packing_algorithm: packing::PackingAlgo,
) -> Vec<Vec<usize>> {
    // this replicates the behavior of the original code
    // all_seq_lens = []
    // for i, count in enumerate(histogram):
    // all_seq_lens.extend([i] * count)
    let all_seq_lens: Vec<usize> = histogram
        .iter()
        .enumerate()
        .flat_map(|(i, &count)| std::iter::repeat_n(i, count))
        .collect();

    let assignments: Vec<Vec<usize>> = packing_algorithm.pack(all_seq_lens, pack_size);

    assignments
}

/// A Python module implemented in Rust.
#[pymodule]
fn binpack_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fast_pack, m)?)?;
    Ok(())
}
