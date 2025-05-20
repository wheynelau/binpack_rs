use pyo3::prelude::*;
use std::collections::{BTreeMap, HashMap};
use crate::{IFileHandles, ReturnFormat, Sequence};

#[pyclass]
pub struct PyReturnIter {
     pub iter: std::vec::IntoIter<HashMap<String, Sequence>>,
}


#[pymethods]
impl PyReturnIter {
    fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }
    fn __len__(slf: PyRef<Self>) -> usize {
        slf.iter.len()
    }
    fn __next__(mut slf: PyRefMut<Self>) -> Option<HashMap<String, Sequence>> {
        slf.iter.next()
    }
}

// TODO: This should use the similar pattern as the `composer_packing_strategy` function
pub(super) fn iterator_packing_strategy(
    ifile_handles: &mut IFileHandles,
    assignments: Vec<Vec<usize>>,
    pack_size: usize,
    pad_id: Option<u32>,
) -> ReturnFormat {
    let mut input_ids = BTreeMap::new();
    let mut positions_ids = BTreeMap::new();

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
    let result: Vec<HashMap<String, Sequence>> = input_ids
        .into_iter()
        .zip(positions_ids)
        .map(|((_, input), (_, positions))| {
            let mut hm = HashMap::new();
            hm.insert("tokens".to_string(), input);
            hm.insert("position_ids".to_string(), positions);
            hm
        })
        .collect();

    let iter = PyReturnIter {
        iter: result.into_iter(),
    };
    ReturnFormat::Iterator(iter)
}
