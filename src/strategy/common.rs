use super::composer::composer_packing_strategy;
use super::nemo::nemo_packing_strategy;
use super::iterator::iterator_packing_strategy;
use crate::NemoOptions;
use crate::{Histogram, IFileHandles, ReturnFormat, Sequence};
use rand::prelude::*;
use std::collections::HashMap;

fn create_position_ids(input_ids: &[Sequence]) -> Vec<Sequence> {
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

pub fn fill_packing_strategy(
    assignments: Vec<Vec<usize>>,
    sequences: Histogram,
    pack_size: usize,
    pad_id: Option<u32>,
    return_format: ReturnFormat,
    options: Option<NemoOptions>,
) -> ReturnFormat {
    let mut ifile_handles: IFileHandles = HashMap::new();
    // Populate the ifile_handles with shuffled input_ids and positions_ids
    populate_ifile_handles(&mut ifile_handles, &sequences, &pack_size);

    // Create the packing strategy
    match return_format {
        ReturnFormat::Nemo(_) => {
            let options = options.expect("PackingOptions is required for Nemo");
            nemo_packing_strategy(&mut ifile_handles, assignments, options, pad_id)
        }
        ReturnFormat::Composer(_) => {
            composer_packing_strategy(&mut ifile_handles, assignments, pack_size, pad_id)
        }
        ReturnFormat::Iterator(_) => {
            iterator_packing_strategy(&mut ifile_handles, assignments, pack_size, pad_id)
        }
    }
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
}
