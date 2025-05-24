// Types and common enums should be defined here
use std::collections::HashMap;

// Sequence usually refers to things like input_ids, position_ids, etc.
pub type Sequence = Vec<u32>;
pub type LossMask = Vec<bool>;

// Histogram is a mapping of sequence lengths to their corresponding sequences
// The key is the length of the sequence, and the value is a vector of dictionaries
// where each dictionary contains the sequence data.
pub type Histogram = HashMap<usize, Vec<HashMap<String, Sequence>>>;

// ifile handles can be adjusted here, but it contains the input_ids and position_ids
pub type IFileHandles = HashMap<usize, (Vec<Sequence>, Vec<Sequence>)>;
