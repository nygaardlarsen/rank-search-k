-- Inputs match benchmarks/data.py: one-based ranks, segment lengths, values.
-- Generate the dataset first:
--   python3 benchmarks/data.py --m 1024 --n 4096 --distribution irregular \
--     --k-mode random --value-mode small --seed 42 --futhark
-- ==
-- nobench input { [1i32] [1i32] [42u32] }
-- output { [42u32] }
-- nobench input { [1i32, 2i32, 3i32] [3i32, 3i32, 3i32]
--                 [3u32, 1u32, 2u32, 3u32, 1u32, 2u32, 3u32, 1u32, 2u32] }
-- output { [1u32, 2u32, 3u32] }
-- nobench input { [3i32, 1i32, 2i32] [5i32, 1i32, 3i32]
--                 [7u32, 2u32, 7u32, 9u32, 7u32, 42u32, 8u32, 0u32, 4u32] }
-- output { [7u32, 42u32, 4u32] }
-- nobench input { [2i32, 3i32] [4i32, 3i32]
--                 [5u32, 5u32, 5u32, 5u32, 0u32, 4294967295u32, 0u32] }
-- output { [5u32, 4294967295u32] }
-- nobench input { [3i32, 2i32] [5i32, 4i32]
--                 [1u32, 2u32, 3u32, 4u32, 5u32, 9u32, 8u32, 7u32, 6u32] }
-- output { [3u32, 7u32] }
-- nobench input { [1i32, 2i32] [3i32, 3i32]
--                 [4294967295u32, 0u32, 2147483648u32, 0u32, 4294967295u32, 2147483648u32] }
-- output { [0u32, 2147483648u32] }
-- compiled input @ benchmarks/datasets/m1024-n4096-irregular-random-small-s42.in

-- Pivot rules. Solved segments have length zero and must not read a pivot.

-- First element of each segment.
def pivot_first [m] [l] (shp: [m]i32) (a: [l]u32) : [m]u32 =
  let ends = scan (+) 0i64 (map i64.i32 shp)
  in map2 (\s end ->
       let s = i64.i32 s
       in if s == 0 then 0u32 else a[end - s]) shp ends

-- Median of the first three elements of each segment (the last element
-- if the segment is shorter than three).
def pivot_median3 [m] [l] (shp: [m]i32) (a: [l]u32) : [m]u32 =
  let ends = scan (+) 0i64 (map i64.i32 shp)
  in map2 (\s end ->
       let s = i64.i32 s
       in if s == 0 then 0u32
          else if s < 3 then a[end - 1]
          else let x = a[end - s]
               let y = a[end - s + 1]
               let z = a[end - s + 2]
               in u32.max (u32.min x y) (u32.min (u32.max x y) z)) shp ends

entry main [m] [n] (ks: [m]i32) (shp: [m]i32) (values: [n]u32) : [m]u32 =
  let lengths = map i64.i32 shp
  let valid_segments = all (\s -> s > 0) shp
  let valid_ranks = all id (map2 (\k s -> 1 <= k && k <= s) ks shp)
  let values = assert (valid_segments && valid_ranks &&
                       reduce (+) 0i64 lengths == n) values

  -- Mark segment starts and propagate each segment's original index
  -- (i32: the ids travel with every element through the filter).
  let ends = scan (+) 0i64 lengths
  let starts = map2 (-) ends lengths
  let markers = scatter (replicate n 0i32) starts
                        (map (\i -> i32.i64 i + 1) (iota m))
  let ii1 = map (\i -> i - 1) (scan i32.max 0i32 markers)

  let (_, _, _, _, result) =
    loop (ks, shp, ii1, a, result) = (ks, shp, ii1, values, replicate m 0u32)
    while length a > 0 do
      -- Pivot: median of the first three elements of each segment.
      let pivots = pivot_median3 shp a

      -- Count the two relevant classes with one histogram over 2m bins:
      -- bin 2i counts "less" of segment i, bin 2i+1 counts "equal";
      -- greater elements get index -1 and are skipped.
      let bins = map2 (\i x ->
        let i = i64.i32 i
        in if x < pivots[i] then 2 * i
           else if x == pivots[i] then 2 * i + 1
           else -1) ii1 a
      let counts = hist (+) 0i32 (2 * m) bins (map (\_ -> 1i32) bins)
      let less = map (\i -> counts[2 * i]) (iota m)
      let equal = map (\i -> counts[2 * i + 1]) (iota m)

      let kinds = map (\i ->
        if shp[i] == 0 then -1i32
        else if ks[i] <= less[i] then 0i32
        else if ks[i] <= less[i] + equal[i] then 1i32
        else 2i32) (iota m)

      let result = map (\i -> if kinds[i] == 1 then pivots[i] else result[i])
                       (iota m)

      let next_ks = map (\i ->
        if kinds[i] == 0 then ks[i]
        else if kinds[i] == 2 then ks[i] - less[i] - equal[i]
        else -1i32) (iota m)
      let next_shp = map (\i ->
        if kinds[i] == 0 then less[i]
        else if kinds[i] == 2 then shp[i] - less[i] - equal[i]
        else 0i32) (iota m)

      -- One stable global filter keeps values and segment IDs aligned.
      let remaining = filter (\(i, x) ->
        (kinds[i] == 0 && x < pivots[i]) ||
        (kinds[i] == 2 && x > pivots[i])) (zip ii1 a)
      let (next_ii1, next_a) = unzip remaining
      in (next_ks, next_shp, next_ii1, next_a, result)
  in result


-- ---------------------------------------------------------------------------
-- Previous rank_search_v2.fut (branch alg_v2), kept for reference.
-- ---------------------------------------------------------------------------

-- -- Inputs match benchmarks/data.py: one-based ranks, segment lengths, values.
-- -- ==
-- -- nobench input { [1i32] [1i32] [42u32] }
-- -- output { [42u32] }
-- -- nobench input { [1i32, 2i32, 3i32] [3i32, 3i32, 3i32]
-- --                 [3u32, 1u32, 2u32, 3u32, 1u32, 2u32, 3u32, 1u32, 2u32] }
-- -- output { [1u32, 2u32, 3u32] }
-- -- nobench input { [3i32, 1i32, 2i32] [5i32, 1i32, 3i32]
-- --                 [7u32, 2u32, 7u32, 9u32, 7u32, 42u32, 8u32, 0u32, 4u32] }
-- -- output { [7u32, 42u32, 4u32] }
-- -- nobench input { [2i32, 3i32] [4i32, 3i32]
-- --                 [5u32, 5u32, 5u32, 5u32, 0u32, 4294967295u32, 0u32] }
-- -- output { [5u32, 4294967295u32] }
-- -- nobench input { [3i32, 2i32] [5i32, 4i32]
-- --                 [1u32, 2u32, 3u32, 4u32, 5u32, 9u32, 8u32, 7u32, 6u32] }
-- -- output { [3u32, 7u32] }
-- -- nobench input { [1i32, 2i32] [3i32, 3i32]
-- --                 [4294967295u32, 0u32, 2147483648u32, 0u32, 4294967295u32, 2147483648u32] }
-- -- output { [0u32, 2147483648u32] }
-- -- compiled input @ data/m1024-n4096-irregular-random-small-s42.in
--
--
-- import "lib/github.com/diku-dk/segmented/segmented"
--
-- def tuple_addition (a: (i32, i32)) (b: (i32, i32)) : (i32, i32) =
--   (a.0 + b.0, a.1 + b.1)
--
-- def mk_flags [m] (n: i64) (lens: [m]i64) : [n]bool =
--   let starts = map2 (-) (scan (+) 0 lens) lens
--   in scatter (replicate n false) starts (replicate m true)
--
-- -- Per-segment logic, on plain values for one segment.
-- def classify (s: i64) (k: i32) (less: i32) (equal: i32) : i32 =
--   if s == 0
--   then -1i32
--   else if k <= less
--   then 0i32
--   else if k <= less + equal
--   then 1i32
--   else 2i32
--
-- def next_rank (kind: i32) (k: i32) (less: i32) (equal: i32) : i32 =
--   if kind == 0
--   then k
--   else if kind == 2
--   then k - less - equal
--   else -1i32
--
-- def next_len (kind: i32) (s: i32) (less: i32) (equal: i32) : i32 =
--   if kind == 0
--   then less
--   else if kind == 2
--   then s - less - equal
--   else 0i32
--
-- entry main [m] [n] (ks: [m]i32) (shp: [m]i32) (values: [n]u32) : [m]u32 =
--   let shp = map i64.i32 shp
--   let ends = scan (+) 0i64 shp 
--   let starts = map2 (-) ends shp
--   let markers = scatter (replicate n 0i64) starts (iota m)
--   let ii1 = scan i64.max 0i64 markers
--   let (_, _, _, _, result) =
--     loop (ks, shp, ii1, a, result) = (ks, shp, ii1, values, replicate m 0u32)
--     while length a > 0 do
--       -- Solved segments have length zero and must not read a pivot.
--       let [l] (a: [l]u32) = a
--       let ii1 = ii1 :> [l]i64
--       let ends = scan (+) 0i64 shp
--       let pivots =
--         map2 (\s end -> if s > 0 then a[end - 1] else 0u32)
--              shp
--              ends
--       -- (sign (x - pivot), sign (x - pivot) ** 2), via comparisons so it cannot overflow
--       let sign_map =
--         map2 (\i x ->
--                 let sgn = i32.bool (x > pivots[i]) - i32.bool (x < pivots[i])
--                 in (sgn, sgn * sgn))
--              ii1
--              a
--       -- One (s, q) total per non-empty segment; empty segments have no start
--       -- flag, so scatter the totals back to their segment ids.
--       let totals = segmented_reduce tuple_addition (0i32, 0i32) (mk_flags l shp) sign_map
--       let [k] (nonempty: [k]i64) = filter (\i -> shp[i] > 0) (iota m)
--       let totals = scatter (replicate m (0i32, 0i32)) nonempty (totals :> [k](i32, i32))
--
--       let counts =
--         map2 (\(s, q) len ->
--                 let less = (q - s) / 2
--                 let equal = i32.i64 len - q
--                 in (less, equal))
--              totals
--              shp
--
--       let (kinds, result, next_ks, next_shp) =
--         map (\i ->
--                let (less, equal) = counts[i]
--                let kind = classify shp[i] ks[i] less equal
--                in ( kind
--                   , if kind == 1 then pivots[i] else result[i]
--                   , next_rank kind ks[i] less equal
--                   , i64.i32 (next_len kind (i32.i64 shp[i]) less equal)
--                   ))
--             (iota m)
--         |> unzip4
--       -- One stable global filter keeps values and segment IDs aligned.
--       let remaining =
--         filter (\(i, x) ->
--                   (kinds[i] == 0 && x < pivots[i])
--                   || (kinds[i] == 2 && x > pivots[i]))
--                (zip ii1 a)
--       let (next_ii1, next_a) = unzip remaining
--       in (next_ks, next_shp, next_ii1, next_a, result)
--   in result
