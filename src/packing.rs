use std::collections::BTreeMap;

use rand::prelude::*;
pub enum PackingAlgo {
    FirstFit,
    FirstFitShuffle,
    FirstFitDecreasing,
}

impl PackingAlgo {
    pub fn pack(&self, seqlens: Vec<usize>, pack_size: usize) -> Vec<Vec<usize>> {
        match self {
            PackingAlgo::FirstFit => first_fit(seqlens, pack_size),
            PackingAlgo::FirstFitShuffle => first_fit_shuffle(seqlens, pack_size),
            PackingAlgo::FirstFitDecreasing => first_fit_decreasing(seqlens, pack_size),
        }
    }
}
impl std::str::FromStr for PackingAlgo {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "first_fit" => Ok(PackingAlgo::FirstFit),
            "first_fit_shuffle" => Ok(PackingAlgo::FirstFitShuffle),
            "first_fit_decreasing" => Ok(PackingAlgo::FirstFitDecreasing),
            _ => Err("Invalid packing algorithm"),
        }
    }
}

fn first_fit(seqlens: Vec<usize>, pack_size: usize) -> Vec<Vec<usize>> {
    let mut res: Vec<Vec<usize>> = Vec::new(); // Holds the packed bins
                                               // Map from remaining capacity to bin indices (ordered)
    let mut capacity_map: BTreeMap<usize, Vec<usize>> = BTreeMap::new();

    for s in seqlens {
        // Find a bin with sufficient capacity
        let mut bin_found = false;
        let mut bin_to_update = None;

        // Find all bins with at least s remaining capacity
        for (&capacity, indices) in capacity_map.range_mut(s..) {
            if !indices.is_empty() {
                // Take the first bin index (maintains first-fit order)
                let bin_idx = indices[0];
                bin_to_update = Some((bin_idx, capacity));
                bin_found = true;
                break;
            }
        }

        if bin_found {
            let (bin_idx, old_capacity) = bin_to_update.unwrap();

            // Remove bin from its current capacity entry
            let indices = capacity_map.get_mut(&old_capacity).unwrap();
            indices.remove(0);
            if indices.is_empty() {
                capacity_map.remove(&old_capacity);
            }

            // Add item to bin
            res[bin_idx].push(s);

            // Update bin's capacity in the map
            let new_capacity = old_capacity - s;
            capacity_map.entry(new_capacity).or_default().push(bin_idx);
        } else {
            // Create a new bin
            let new_bin_idx = res.len();
            res.push(vec![s]);
            let remaining = pack_size - s;
            capacity_map.entry(remaining).or_default().push(new_bin_idx);
        }
    }

    res
}

fn first_fit_decreasing(seqlens: Vec<usize>, pack_size: usize) -> Vec<Vec<usize>> {
    let mut seqlens = seqlens;
    seqlens.sort_by(|a, b| b.cmp(a));
    first_fit(seqlens, pack_size)
}
// Shuffle won't be tested
fn first_fit_shuffle(seqlens: Vec<usize>, pack_size: usize) -> Vec<Vec<usize>> {
    let mut seqlens = seqlens;
    let mut rng = rand::rng();
    seqlens.shuffle(&mut rng);
    first_fit(seqlens, pack_size)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_first_fit() {
        let seqlens = vec![1, 2, 3, 4, 5];
        let pack_size = 5;
        let result = first_fit(seqlens.clone(), pack_size);
        dbg!(&result);
        assert_eq!(result.len(), 4);
        assert_eq!(result[0], vec![1, 2]);
        assert_eq!(result[1], vec![3]);
        assert_eq!(result[2], vec![4]);
        assert_eq!(result[3], vec![5]);
    }

    #[test]
    fn test_first_fit_decreasing() {
        let seqlens = vec![1, 2, 3, 4, 5];
        let pack_size = 5;
        let result = first_fit_decreasing(seqlens.clone(), pack_size);
        dbg!(&result);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], vec![5]);
        assert_eq!(result[1], vec![4, 1]);
        assert_eq!(result[2], vec![3, 2]);
    }
}
