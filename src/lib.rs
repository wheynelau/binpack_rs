use pyo3::prelude::*;
use rand::prelude::*;
use std::collections::HashMap;

pub mod packing;

#[derive(IntoPyObject, IntoPyObjectRef)]
enum ReturnFormat {
    Composer(HashMap<String, Vec<Vec<u32>>>),
}
#[allow(dead_code)]
enum InputFormat {
    DictOfList(HashMap<String, Vec<Sequence>>),
    ListOfDicts(Vec<HashMap<String, Sequence>>),
}
// Sequence usually refers to things like input_ids, position_ids, etc.

type Sequence = Vec<u32>;
type Histogram = HashMap<usize, Vec<HashMap<String, Sequence>>>;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn fast_pack(
    examples: HashMap<String, Vec<Sequence>>,
    target_pack_size: usize,
    packing_algorithm: String,
    pad_id: Option<u32>,
) -> PyResult<ReturnFormat> {
    let (sequences, seq_lens) = create_hist(examples.clone(), target_pack_size);
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
    let result = fill_packing_strategy(assignments, sequences, target_pack_size, pad_id);
    Ok(result)
}

fn create_hist(
    dataset: HashMap<String, Vec<Sequence>>,
    truncate_seq_len: usize,
) -> (Histogram, Vec<usize>) {
    let mut sequences: HashMap<usize, Vec<HashMap<String, Sequence>>> = HashMap::new();
    let mut counts = vec![0u32; truncate_seq_len + 1];
    let mut seq_lens: Vec<usize> = Vec::new();

    // format the input data into a list of dicts
    let dataset = dataset
        .into_iter()
        .flat_map(|(key, value)| {
            value.into_iter().map(move |v| {
                let mut entry = HashMap::new();
                entry.insert(key.clone(), v);
                entry
            })
        })
        .collect::<Vec<_>>();

    dataset.into_iter().for_each(|entry| {
        // Only need input_ids key
        let seq = entry
            .get("input_ids")
            .expect("Expected key 'input_ids' in the dataset entry");
        let seq_len = seq.len();
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
        .flat_map(|(i, &count)| std::iter::repeat(i).take(count))
        .collect();

    let assignments: Vec<Vec<usize>> = packing_algorithm.pack(all_seq_lens, pack_size);
    // let packed_seq_lens: Vec<usize> = assignments
    //     .iter()
    //     .map(|bin| bin.iter().sum())
    //     .collect();

    // let packing_factor =
    assignments
}

fn fill_packing_strategy(
    assignments: Vec<Vec<usize>>,
    sequences: HashMap<usize, Vec<HashMap<String, Sequence>>>,
    pack_size: usize,
    pad_id: Option<u32>,
) -> ReturnFormat {
    let mut ifile_handles: HashMap<usize, (Vec<Sequence>, Vec<Sequence>)> = HashMap::new();
    let mut rng = rand::rng();
    for seq_len in 0..(pack_size + 1) {
        // Try to replicate python behavior
        let per_seq_data = sequences.get(&seq_len);
        let per_seq_len = per_seq_data.map_or(0, |v| v.len());
        if per_seq_len > 0 {
            let mut input_ids = per_seq_data
                .unwrap() // can be safely unwrapped, since we checked above
                .iter()
                .map(|entry| {
                    entry
                        .get("input_ids")
                        .expect("Expected key 'input_ids' in the dataset entry")
                        .clone()
                })
                .collect::<Vec<Sequence>>();
            // shuffle the input_ids
            input_ids.shuffle(&mut rng);

            let positions_ids = input_ids
                .iter()
                .map(|seq| {
                    let mut pos = vec![0u32; seq.len()];
                    // Non idiomatic way kept for reference
                    // for i in 0..seq.len() {
                    //     pos[i] = seq[i];
                    // }
                    pos[..seq.len()].copy_from_slice(&seq[..]);
                    pos
                })
                .collect::<Vec<Sequence>>();

            ifile_handles.insert(seq_len, (input_ids, positions_ids));
        }
    }

    let mut input_ids = HashMap::new();
    let mut positions_ids = HashMap::new();

    for (oindex, assignment) in assignments.iter().enumerate() {
        let mut _input_ids: Sequence = Vec::new();
        let mut _positions_ids: Sequence = Vec::new();
        for seq_len in assignment {
            if let Some((input_ids_vec, positions_ids_vec)) = ifile_handles.get_mut(seq_len) {
                _input_ids.extend(
                    input_ids_vec
                        .pop()
                        .expect("Expected input_ids to be available"),
                );
                _positions_ids.extend(
                    positions_ids_vec
                        .pop()
                        .expect("Expected positions_ids to be available"),
                );
            }
        }

        // Handle padding and truncation here
        if _input_ids.len() > pack_size {
            _input_ids.truncate(pack_size);
            _positions_ids.truncate(pack_size);
        } else if let Some(pad_id) = pad_id {
            let pad_len = pack_size - _input_ids.len();
            _input_ids.extend(vec![pad_id; pad_len]);
            _positions_ids.extend(vec![0; pad_len]); // position ids are all 0
        }
        input_ids.insert(oindex, _input_ids);
        positions_ids.insert(oindex, _positions_ids);
    }
    // Here handle the conversion to the desired format
    // for now is only composer format, which is a vec
    let list_input_ids: Vec<Sequence> = input_ids.values().cloned().collect();
    let _list_positions_ids: Vec<Sequence> = positions_ids.values().cloned().collect();
    let mut result = HashMap::new();
    result.insert("tokens".to_string(), list_input_ids);
    ReturnFormat::Composer(result)
}

/// A Python module implemented in Rust.
#[pymodule]
fn binpack_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fast_pack, m)?)?;
    Ok(())
}
