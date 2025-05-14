use pyo3::prelude::*;
use rand::prelude::*;
use std::{collections::HashMap, vec};

pub mod packing;

#[derive(IntoPyObject, IntoPyObjectRef)]
enum ReturnFormat {
    Composer(HashMap<String, Vec<Vec<u32>>>),
    // Nemo has the same format, but the keys are different
    // Different entries
    Nemo(HashMap<String, Vec<Vec<u32>>>),
}
impl std::str::FromStr for ReturnFormat {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "nemo" => Ok(ReturnFormat::Nemo(HashMap::new())),
            "composer" => Ok(ReturnFormat::Composer(HashMap::new())),
            _ => Err("Invalid return format"),
        }
    }
}

#[allow(dead_code)]
enum InputFormat {
    DictOfList(HashMap<String, Vec<Sequence>>),
    ListOfDicts(Vec<HashMap<String, Sequence>>),
}
// Sequence usually refers to things like input_ids, position_ids, etc.
type Sequence = Vec<u32>;

// Histogram is a mapping of sequence lengths to their corresponding sequences
// The key is the length of the sequence, and the value is a vector of dictionaries
// where each dictionary contains the sequence data.
type Histogram = HashMap<usize, Vec<HashMap<String, Sequence>>>;

// ifile handles can be adjusted here, but it contains the input_ids and position_ids
type IFileHandles = HashMap<usize, (Vec<Sequence>, Vec<Sequence>)>;

/// Formats the sum of two numbers as string.
#[pyfunction]
#[pyo3(signature = (examples, target_pack_size, packing_algorithm, return_format, pad_id))]
fn fast_pack(
    examples: HashMap<String, Vec<Sequence>>,
    target_pack_size: usize,
    packing_algorithm: String,
    return_format: String,
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
    let return_format = match return_format.parse::<ReturnFormat>() {
        Ok(return_format) => return_format,
        Err(_) => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Invalid return format. Use 'nemo' or 'composer'.",
            ))
        }
    };
    let assignments = create_packing_strategy(seq_lens, target_pack_size, packing_algorithm);
    let result = fill_packing_strategy(
        assignments,
        sequences,
        target_pack_size,
        pad_id,
        return_format,
    );
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
        .flat_map(|(i, &count)| std::iter::repeat(i).take(count))
        .collect();

    let assignments: Vec<Vec<usize>> = packing_algorithm.pack(all_seq_lens, pack_size);

    assignments
}

fn create_position_ids(input_ids: &Vec<Sequence>) -> Vec<Sequence> {
    // Create position ids based on the input_ids
    let positions_ids = input_ids
        .iter()
        .map(|seq| {
            let pos = seq
                .iter()
                .enumerate()
                .map(|(i, _)| i as u32)
                .collect::<Sequence>();
            pos
        })
        .collect::<Vec<Sequence>>();
    positions_ids
}
// Note that nemo has a different implementation, their answer_start_idx refers to the
// start of the answer, while here, we use the idx that's before the answer, usually something like
// the assistant message
fn create_loss_mask(
    input_ids: Sequence,
    answer_loss_only: bool,
    answer_start_id: Option<u32>,
    answer_end_id: Option<u32>,
    pad_id: Option<u32>,
) -> Sequence {
    // If answer_loss_only is false, return a mask of ones
    if !answer_loss_only {
        let mut loss_mask = vec![1; input_ids.len()];
        loss_mask[0] = 0; // The first token is always 0
        return loss_mask;
    }
    // Otherwise, create a mask based on the answer_start_id and answer_end_id
    let mut loss_mask: Sequence = vec![0; input_ids.len()];
    // unwrap idx here
    let answer_start_id = answer_start_id.expect("answer_start_id is None");
    let answer_end_id = answer_end_id.expect("answer_end_id is None");
    // logic here is the default is 0, when the answer starts, the flag is 1, until the answer ends
    let mut is_answer = false;
    for i in 0..input_ids.len() {
        if let Some(pad_id) = pad_id {
            if input_ids[i] == pad_id {
                loss_mask[i] = 0;
                continue;
            }
        } // The next few checks would not be possible if pad_id is set
        if input_ids[i] == answer_start_id {
            is_answer = true;
        } else if input_ids[i] == answer_end_id {
            is_answer = false;
        }
        // regardless the answer. if the input is pad_id, set it to 0
        
        loss_mask[i] = if is_answer { 1 } else { 0 };
    }
    loss_mask
}

fn populate_ifile_handles(
    ifile_handles: &mut IFileHandles,
    sequences: &Histogram,
    pack_size: &usize,
) {
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

            let position_ids = create_position_ids(&input_ids);

            ifile_handles.insert(seq_len, (input_ids, position_ids));
        }
    }
}

fn nemo_packing_strategy(
    ifile_handles: &mut IFileHandles,
    assignments: Vec<Vec<usize>>,
    answer_start_id: Option<u32>,
    answer_end_id: Option<u32>,
    answer_loss_only: bool,
    pad_id: Option<u32>,
) -> ReturnFormat {
    // Similar to fill_packing_strategy but for Nemo format
    // This is a placeholder for the actual implementation
    let mut input_ids = HashMap::new();
    let mut loss_mask = HashMap::new();
    let mut seq_start_id = HashMap::new();

    assignments
        .iter()
        .enumerate()
        .for_each(|(oindex, assignment)| {
            let mut _input_ids: Sequence = Vec::new();
            // Loss mask only needs 0,1 but for easier conversion, use u32
            let mut _loss_mask: Sequence = Vec::new();
            let mut _seq_start_id: Sequence = vec![0];
            for seq_len in assignment {
                if let Some((input_ids_vec, positions_ids_vec)) = ifile_handles.get_mut(seq_len) {
                    let _input_vec: Sequence = input_ids_vec
                        .pop()
                        .expect("Expected input_ids to be available");
                    _input_ids.extend(_input_vec.clone());
                    let loss_mask = create_loss_mask(
                        _input_vec,
                        answer_loss_only,
                        answer_start_id,
                        answer_end_id,
                        pad_id,
                    );
                    _loss_mask.extend(loss_mask);
                    _ = positions_ids_vec // positions_ids are not used in Nemo, but still need to be popped
                        .pop()
                        .expect("Expected positions_ids to be available")
                }
            } // Loop handling assignment ends here
            input_ids.insert(oindex, _input_ids);
            loss_mask.insert(oindex, _loss_mask);
            // in the python implementation, a slice up to -1 is used
            // but i didn't see a need that this variable is used
            // so i just pop the last element
            _seq_start_id.pop();
            seq_start_id.insert(oindex, _seq_start_id);
        }); // for each ends here
            // for the return format
    let list_input_ids: Vec<Sequence> = input_ids.values().cloned().collect();
    let list_position_ids: Vec<Sequence> = loss_mask.values().cloned().collect();
    let list_seq_start_id: Vec<Sequence> = seq_start_id.values().cloned().collect();
    let mut result = HashMap::new();
    result.insert("input_ids".to_string(), list_input_ids);
    result.insert("loss_mask".to_string(), list_position_ids);
    result.insert("seq_start_id".to_string(), list_seq_start_id);

    ReturnFormat::Nemo(result)
}

fn composer_packing_strategy(
    ifile_handles: &mut IFileHandles,
    assignments: Vec<Vec<usize>>,
    pack_size: usize,
    pad_id: Option<u32>,
) -> ReturnFormat {
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
    let list_positions_ids: Vec<Sequence> = positions_ids.values().cloned().collect();
    let mut result = HashMap::new();
    result.insert("tokens".to_string(), list_input_ids);
    result.insert("positions_ids".to_string(), list_positions_ids);
    ReturnFormat::Composer(result)
}
fn fill_packing_strategy(
    assignments: Vec<Vec<usize>>,
    sequences: Histogram,
    pack_size: usize,
    pad_id: Option<u32>,
    return_format: ReturnFormat,
) -> ReturnFormat {
    let mut ifile_handles: IFileHandles = HashMap::new();
    // Populate the ifile_handles with shuffled input_ids and positions_ids
    populate_ifile_handles(&mut ifile_handles, &sequences, &pack_size);

    // Create the packing strategy
    match return_format {
        ReturnFormat::Nemo(_) => {
            nemo_packing_strategy(&mut ifile_handles, assignments, None, None, false, pad_id)
        }
        ReturnFormat::Composer(_) => {
            composer_packing_strategy(&mut ifile_handles, assignments, pack_size, pad_id)
        }
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn binpack_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fast_pack, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_ids() {
        let input_ids = vec![vec![1, 2, 3], vec![4, 5, 6, 7]];
        let position_ids = create_position_ids(&input_ids);
        assert_eq!(position_ids[0], vec![0, 1, 2]);
        assert_eq!(position_ids[1], vec![0, 1, 2, 3]);
    }
    #[test]
    fn test_loss_mask() {
        // No answer
        let input_ids = vec![1,2,3,4,5];
        let loss_mask = create_loss_mask(input_ids, false, None, None, None);
        assert_eq!(loss_mask, vec![0, 1, 1, 1, 1]);
        let input_ids = vec![2, 105, 2364, 107, 3689, 563, 506, 5279, 529, 7001, 236881, 106, 107, 105, 4368, 107, 818, 5279, 529, 7001, 563, 9079, 236761, 106, 107, 105, 2364, 107, 3689, 563, 506, 5279, 529, 9405, 236881, 106, 107, 105, 4368, 107, 818, 5279, 529, 9405, 563, 15687, 236761, 106, 107];
        let answer_start_id = Some(4368);
        let answer_end_id = Some(106);
        let pad_id = None;
        // One way to think of loss mask is like setting -100 for labels
        // that are not in the answer
        let loss_mask = create_loss_mask(input_ids, true, answer_start_id, answer_end_id, pad_id);
        assert_eq!(loss_mask, vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]);
        let pad_id = Some(5279);
        // Inject a pad id to test
        let input_ids = vec![2, 105, 2364, 107, 3689, 563, 506, 5279, 529, 7001, 236881, 106, 107, 105, 4368, 107, 818, 5279, 529, 7001, 563, 9079, 236761, 106, 107, 105, 2364, 107, 3689, 563, 506, 5279, 529, 9405, 236881, 106, 107, 105, 4368, 107, 818, 5279, 529, 9405, 563, 0, 236761, 106, 107];
        let loss_mask = create_loss_mask(input_ids, true, answer_start_id, answer_end_id, pad_id);
        assert_eq!(loss_mask, vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0]);
    }
}
