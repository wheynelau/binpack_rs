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
        match s {
            "first_fit" => Ok(PackingAlgo::FirstFit),
            "first_fit_shuffle" => Ok(PackingAlgo::FirstFitShuffle),
            "first_fit_decreasing" => Ok(PackingAlgo::FirstFitDecreasing),
            _ => Err("Invalid packing algorithm"),
        }
    }
}

fn first_fit(seqlens: Vec<usize>, pack_size: usize) -> Vec<Vec<usize>> {
    let mut res: Vec<Vec<usize>> = Vec::new(); // Holds the packed bins
    let mut sum_of_bin: Vec<usize> = Vec::new(); // Holds the sum of each bin
    'outer: for s in seqlens {
        for i in 0..res.len() {
            if sum_of_bin[i] + s <= pack_size {
                res[i].push(s);
                sum_of_bin[i] += s;
                continue 'outer;
            }
        }
        // If no bin fits, create a new one
        res.push(vec![s]);
        sum_of_bin.push(s);
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