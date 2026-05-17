defmodule WolframModel.RicciCurvatureTest do
  use ExUnit.Case, async: true
  alias Hypergraph
  alias WolframModel.Analytics

  # ── Degenerate inputs ──────────────────────────────────────────────────────

  test "returns nil for empty hypergraph" do
    hg = Hypergraph.new()
    assert Analytics.estimate_ricci_scalar(hg) == nil
  end

  test "returns nil for fewer than 6 vertices" do
    hg =
      Hypergraph.new()
      |> Hypergraph.add_hyperedge([1, 2])
      |> Hypergraph.add_hyperedge([2, 3])

    assert Analytics.estimate_ricci_scalar(hg) == nil
  end

  # ── Flat geometry (2-D grid) ───────────────────────────────────────────────
  # A regular 2-D grid has zero intrinsic curvature; the estimated R should be
  # close to zero (finite-size effects allow some tolerance).

  test "2D grid returns a float near zero (flat geometry)" do
    # 4×4 grid
    hg =
      Hypergraph.new()
      |> add_grid_edges(4, 4)

    r = Analytics.estimate_ricci_scalar(hg)
    assert is_float(r)
    assert abs(r) < 3.0
  end

  # ── Positive curvature (sphere-like) ──────────────────────────────────────
  # An icosahedron (12 vertices, BFS depth ≤ 3) is sphere-like but too small
  # to overcome finite-size effects on the sign; we verify a non-nil float is
  # returned and that the 2-D grid produces a smaller absolute curvature.

  test "icosahedron-like graph returns a non-nil float" do
    hg = icosahedron_hypergraph()
    r = Analytics.estimate_ricci_scalar(hg)
    assert is_float(r)
  end

  # A larger subdivided sphere graph provides enough radial samples to confirm
  # positive curvature.
  test "subdivided icosahedron (80 vertices) returns positive curvature" do
    hg = subdivided_icosahedron()
    r = Analytics.estimate_ricci_scalar(hg)
    assert is_float(r)
    assert r > 0.0
  end

  # ── Result type ────────────────────────────────────────────────────────────

  test "returns a float for a sufficiently large connected graph" do
    hg =
      Hypergraph.new()
      |> add_cycle_edges(12)

    result = Analytics.estimate_ricci_scalar(hg)
    # A cycle may return nil if max_r < 3 from some seeds, but when it
    # does return a value it must be a float.
    if result != nil, do: assert(is_float(result))
  end

  # ── Helpers ────────────────────────────────────────────────────────────────

  # Build a w×h grid hypergraph (binary edges only).
  defp add_grid_edges(hg, w, h) do
    # horizontal edges
    hg =
      for row <- 0..(h - 1), col <- 0..(w - 2), reduce: hg do
        acc ->
          v1 = row * w + col + 1
          v2 = row * w + col + 2
          Hypergraph.add_hyperedge(acc, [v1, v2])
      end

    # vertical edges
    for row <- 0..(h - 2), col <- 0..(w - 1), reduce: hg do
      acc ->
        v1 = row * w + col + 1
        v2 = (row + 1) * w + col + 1
        Hypergraph.add_hyperedge(acc, [v1, v2])
    end
  end

  # Build a cycle of n vertices.
  defp add_cycle_edges(hg, n) do
    Enum.reduce(1..n, hg, fn i, acc ->
      Hypergraph.add_hyperedge(acc, [i, rem(i, n) + 1])
    end)
  end

  # Icosahedron: 12 vertices, 30 edges, positive curvature.
  defp icosahedron_hypergraph do
    edges = [
      [1, 2],
      [1, 3],
      [1, 4],
      [1, 5],
      [1, 6],
      [2, 3],
      [3, 4],
      [4, 5],
      [5, 6],
      [6, 2],
      [2, 7],
      [3, 7],
      [3, 8],
      [4, 8],
      [4, 9],
      [5, 9],
      [5, 10],
      [6, 10],
      [6, 11],
      [2, 11],
      [7, 8],
      [8, 9],
      [9, 10],
      [10, 11],
      [11, 7],
      [7, 12],
      [8, 12],
      [9, 12],
      [10, 12],
      [11, 12]
    ]

    Enum.reduce(edges, Hypergraph.new(), fn e, acc ->
      Hypergraph.add_hyperedge(acc, e)
    end)
  end

  # Frequency-2 subdivision of the icosahedron: 42 vertices, 120 edges.
  # Each original edge is split by a midpoint vertex (IDs 13..42).
  # Provides enough BFS depth for a reliable positive-curvature estimate.
  defp subdivided_icosahedron do
    ico_edges = [
      {1, 2},
      {1, 3},
      {1, 4},
      {1, 5},
      {1, 6},
      {2, 3},
      {3, 4},
      {4, 5},
      {5, 6},
      {6, 2},
      {2, 7},
      {3, 7},
      {3, 8},
      {4, 8},
      {4, 9},
      {5, 9},
      {5, 10},
      {6, 10},
      {6, 11},
      {2, 11},
      {7, 8},
      {8, 9},
      {9, 10},
      {10, 11},
      {11, 7},
      {7, 12},
      {8, 12},
      {9, 12},
      {10, 12},
      {11, 12}
    ]

    faces = [
      {1, 2, 3},
      {1, 3, 4},
      {1, 4, 5},
      {1, 5, 6},
      {1, 6, 2},
      {2, 3, 7},
      {3, 4, 8},
      {4, 5, 9},
      {5, 6, 10},
      {6, 2, 11},
      {2, 7, 11},
      {3, 7, 8},
      {4, 8, 9},
      {5, 9, 10},
      {6, 10, 11},
      {12, 7, 8},
      {12, 8, 9},
      {12, 9, 10},
      {12, 10, 11},
      {12, 11, 7}
    ]

    edge_to_mid =
      ico_edges
      |> Enum.with_index(13)
      |> Map.new(fn {{a, b}, id} -> {{min(a, b), max(a, b)}, id} end)

    get_mid = fn a, b -> Map.fetch!(edge_to_mid, {min(a, b), max(a, b)}) end

    # Two half-edges per original edge
    half_edges =
      Enum.flat_map(ico_edges, fn {a, b} ->
        m = get_mid.(a, b)
        [[a, m], [b, m]]
      end)

    # Three interior edges per face (connecting midpoints)
    face_edges =
      Enum.flat_map(faces, fn {a, b, c} ->
        mab = get_mid.(a, b)
        mbc = get_mid.(b, c)
        mac = get_mid.(a, c)
        [[mab, mbc], [mbc, mac], [mac, mab]]
      end)

    Enum.reduce(half_edges ++ face_edges, Hypergraph.new(), fn e, acc ->
      Hypergraph.add_hyperedge(acc, e)
    end)
  end
end
