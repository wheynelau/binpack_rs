use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::{IFileHandles, LossMask, NemoFormat, ReturnFormat, Sequence};
use std::collections::HashMap;

pub struct NemoOptions {
    answer_start_id: Option<u32>,
    answer_end_id: Option<u32>,
    answer_loss_only: bool,
}

impl NemoOptions {
    pub fn builder() -> NemoOptionsBuilder {
        NemoOptionsBuilder::default()
    }

    // Validate the options
    fn validate(&self) -> Result<(), String> {
        if self.answer_loss_only && (self.answer_start_id.is_none() || self.answer_end_id.is_none())
        {
            return Err(
                "answer_loss_only is set to true, but answer_start_id or answer_end_id is None"
                    .to_string(),
            );
        }
        Ok(())
    }
}

#[derive(Default)]
pub struct NemoOptionsBuilder {
    answer_start_id: Option<u32>,
    answer_end_id: Option<u32>,
    answer_loss_only: bool,
}

impl NemoOptionsBuilder {
    pub fn answer_start_id(mut self, id: Option<u32>) -> Self {
        self.answer_start_id = id;
        self
    }

    pub fn answer_end_id(mut self, id: Option<u32>) -> Self {
        self.answer_end_id = id;
        self
    }

    pub fn answer_loss_only(mut self, loss_only: bool) -> Self {
        self.answer_loss_only = loss_only;
        self
    }

    pub fn from_py_dict(mut self, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        if let Some(kwargs) = kwargs {
            for (key, value) in kwargs.iter() {
                if let Ok(key_str) = key.extract::<&str>() {
                    match key_str {
                        "answer_start_id" => self.answer_start_id = value.extract().unwrap_or(None),
                        "answer_end_id" => self.answer_end_id = value.extract().unwrap_or(None),
                        "answer_loss_only" => {
                            self.answer_loss_only = value.extract().unwrap_or(false)
                        }
                        _ => continue,
                    }
                }
            }
        }
        Ok(self)
    }

    pub fn build(self) -> PyResult<NemoOptions> {
        let mut options = NemoOptions {
            answer_start_id: self.answer_start_id,
            answer_end_id: self.answer_end_id,
            answer_loss_only: self.answer_loss_only,
        };

        // Apply business logic
        if !options.answer_loss_only {
            options.answer_start_id = None;
            options.answer_end_id = None;
        }

        // Validate the options
        if let Err(msg) = options.validate() {
            return Err(PyValueError::new_err(msg));
        }

        Ok(options)
    }
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
) -> LossMask {
    // If answer_loss_only is false, return a mask of ones
    if !answer_loss_only {
        let mut loss_mask = vec![true; input_ids.len()];
        loss_mask[0] = false; // The first token is always 0
                              // replace pad_id with 0
        if let Some(pad_id) = pad_id {
            for i in 0..input_ids.len() {
                if input_ids[i] == pad_id {
                    loss_mask[i] = false;
                }
            }
        }
        return loss_mask;
    }
    // Otherwise, create a mask based on the answer_start_id and answer_end_id
    let mut loss_mask: LossMask = vec![false; input_ids.len()];
    // unwrap idx here
    let answer_start_id = answer_start_id.expect("answer_start_id is None");
    let answer_end_id = answer_end_id.expect("answer_end_id is None");
    // logic here is the default is 0, when the answer starts, the flag is 1, until the answer ends
    for i in 0..input_ids.len() {
        if let Some(pad_id) = pad_id {
            if input_ids[i] == pad_id {
                loss_mask[i] = false;
                continue;
            }
        } // The next few checks would not be possible if pad_id is set
        if input_ids[i] == answer_start_id {
            loss_mask[i] = true;
        } else if input_ids[i] == answer_end_id {
            loss_mask[i] = false;
        }
    }
    loss_mask
}

pub(super) fn nemo_packing_strategy(
    ifile_handles: &mut IFileHandles,
    assignments: Vec<Vec<usize>>,
    options: NemoOptions,
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
            let mut _loss_mask: LossMask = Vec::new();
            let mut _seq_start_id: Sequence = vec![0];
            for seq_len in assignment {
                if let Some((input_ids_vec, positions_ids_vec)) = ifile_handles.get_mut(seq_len) {
                    let _input_vec: Sequence = input_ids_vec
                        .pop()
                        .expect("Expected input_ids to be available");
                    _input_ids.extend(_input_vec.clone());
                    let loss_mask = create_loss_mask(
                        _input_vec,
                        options.answer_loss_only,
                        options.answer_start_id,
                        options.answer_end_id,
                        pad_id,
                    );
                    _loss_mask.extend(loss_mask);
                    _ = positions_ids_vec // positions_ids are not used in Nemo, but still need to be popped
                        .pop()
                        .expect("Expected positions_ids to be available");
                    _seq_start_id.push(_input_ids.len() as u32);
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
    let list_position_ids: Vec<LossMask> = loss_mask.values().cloned().collect();
    let list_seq_start_id: Vec<Sequence> = seq_start_id.values().cloned().collect();
    let mut result: HashMap<String, NemoFormat> = HashMap::new();
    result.insert("input_ids".to_string(), NemoFormat::Tokens(list_input_ids));
    result.insert(
        "loss_mask".to_string(),
        NemoFormat::LossMask(list_position_ids),
    );
    result.insert(
        "seq_start_id".to_string(),
        NemoFormat::Tokens(list_seq_start_id),
    );

    ReturnFormat::Nemo(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_loss_mask() {
        // No answer
        let input_ids = vec![1, 2, 3, 4, 5];
        let loss_mask = create_loss_mask(input_ids, false, None, None, None);
        assert_eq!(loss_mask, vec![false, true, true, true, true]);
        let input_ids = vec![
            2, 105, 2364, 107, 3689, 563, 506, 5279, 529, 7001, 236881, 106, 107, 105, 4368, 107,
            818, 5279, 529, 7001, 563, 9079, 236761, 106, 107, 105, 2364, 107, 3689, 563, 506,
            5279, 529, 9405, 236881, 106, 107, 105, 4368, 107, 818, 5279, 529, 9405, 563, 15687,
            236761, 106, 107,
        ];
        let answer_start_id = Some(4368);
        let answer_end_id = Some(106);
        let pad_id = None;
        // One way to think of loss mask is like setting -100 for labels
        // that are not in the answer
        let loss_mask = create_loss_mask(input_ids, true, answer_start_id, answer_end_id, pad_id);
        assert_eq!(
            loss_mask,
            vec![
                false, false, false, false, false, false, false, false, false, false, false, false,
                false, false, true, true, true, true, true, true, true, true, true, false, false,
                false, false, false, false, false, false, false, false, false, false, false, false,
                false, true, true, true, true, true, true, true, true, true, false, false
            ]
        );
        let pad_id = Some(5279);
        // Inject a pad id to test
        let input_ids = vec![
            2, 105, 2364, 107, 3689, 563, 506, 5279, 529, 7001, 236881, 106, 107, 105, 4368, 107,
            818, 5279, 529, 7001, 563, 9079, 236761, 106, 107, 105, 2364, 107, 3689, 563, 506,
            5279, 529, 9405, 236881, 106, 107, 105, 4368, 107, 818, 5279, 529, 9405, 563, 15687,
            236761, 106, 107,
        ];
        let loss_mask = create_loss_mask(input_ids, true, answer_start_id, answer_end_id, pad_id);
        assert_eq!(
            loss_mask,
            vec![
                false, false, false, false, false, false, false, false, false, false, false, false,
                false, false, true, true, true, false, true, true, true, true, true, false, false,
                false, false, false, false, false, false, false, false, false, false, false, false,
                false, true, true, true, false, true, true, true, true, true, false, false
            ]
        );
    }
}
