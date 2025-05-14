use crate::{IFileHandles, ReturnFormat, Sequence};
use std::collections::HashMap;

pub(super) fn composer_packing_strategy(
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
