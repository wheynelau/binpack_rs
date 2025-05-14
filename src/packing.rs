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
    // pub fn from_str(s: &str) -> Option<PackingAlgo> {
    //     match s {
    //         "first_fit" => Some(PackingAlgo::FirstFit),
    //         "first_fit_shuffle" => Some(PackingAlgo::FirstFitShuffle),
    //         "first_fit_decreasing" => Some(PackingAlgo::FirstFitDecreasing),
    //         _ => None,
    //     }
    // }
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
    for s in seqlens {
        // find first bin that fits
        for i in 0..res.len() {
            if sum_of_bin[i] + s <= pack_size {
                res[i].push(s);
                sum_of_bin[i] += s;
                break;
            } else {
                res.push(vec![s]);
                sum_of_bin.push(s);
            }
        }
    }
    res
}

fn first_fit_decreasing(seqlens: Vec<usize>, pack_size: usize) -> Vec<Vec<usize>> {
    let mut seqlens = seqlens;
    seqlens.sort_by(|a, b| b.cmp(a));
    first_fit(seqlens, pack_size)
}

fn first_fit_shuffle(seqlens: Vec<usize>, pack_size: usize) -> Vec<Vec<usize>> {
    let mut seqlens = seqlens;
    let mut rng = rand::rng();
    seqlens.shuffle(&mut rng);
    first_fit(seqlens, pack_size)
}
