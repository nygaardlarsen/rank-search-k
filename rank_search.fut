-- Inputs match benchmarks/data.py: one-based ranks, segment lengths, values.
-- ==
-- input { [1i32] [1i32] [42u32] }
-- output { [42u32] }
-- input { [1i32, 2i32, 3i32] [3i32, 3i32, 3i32]
--         [3u32, 1u32, 2u32, 3u32, 1u32, 2u32, 3u32, 1u32, 2u32] }
-- output { [1u32, 2u32, 3u32] }
-- input { [3i32, 1i32, 2i32] [5i32, 1i32, 3i32]
--         [7u32, 2u32, 7u32, 9u32, 7u32, 42u32, 8u32, 0u32, 4u32] }
-- output { [7u32, 42u32, 4u32] }
-- input { [2i32, 3i32] [4i32, 3i32]
--         [5u32, 5u32, 5u32, 5u32, 0u32, 4294967295u32, 0u32] }
-- output { [5u32, 4294967295u32] }
-- input { [3i32, 2i32] [5i32, 4i32]
--         [1u32, 2u32, 3u32, 4u32, 5u32, 9u32, 8u32, 7u32, 6u32] }
-- output { [3u32, 7u32] }

entry main [m] [n] (ks: [m]i32) (shp: [m]i32) (values: [n]u32) : [m]u32 =
  let lengths = map i64.i32 shp
  let valid_segments = all (\s -> s > 0) shp
  let valid_ranks = all id (map2 (\k s -> 1 <= k && k <= s) ks shp)
  let values = assert (valid_segments && valid_ranks &&
                       reduce (+) 0i64 lengths == n) values

  -- Mark segment starts and propagate each segment's original index.
  let ends = scan (+) 0i64 lengths
  let starts = map2 (-) ends lengths
  let markers = scatter (replicate n 0i64) starts
                        (map (\i -> i + 1) (iota m))
  let ii1 = map (\i -> i - 1) (scan i64.max 0i64 markers)

  let (_, _, _, _, result) =
    loop (ks, shp, ii1, a, result) = (ks, shp, ii1, values, replicate m 0u32)
    while length a > 0 do
      -- Solved segments have length zero and must not read a pivot.
      let ends = scan (+) 0i64 (map i64.i32 shp)
      let pivots = map2 (\s end -> if s > 0 then a[end - 1] else 0u32)
                        shp ends

      -- Count the two relevant classes without constructing partitions.
      let less = hist (+) 0i32 m ii1
                      (map2 (\i x -> if x < pivots[i] then 1i32 else 0i32) ii1 a)
      let equal = hist (+) 0i32 m ii1
                       (map2 (\i x -> if x == pivots[i] then 1i32 else 0i32) ii1 a)
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
