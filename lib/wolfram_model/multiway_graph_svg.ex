defmodule WolframModel.MultiwayGraphSVG do
  @moduledoc """
  SVG rendering of the multiway evolution graph (DAG or tree).

  Accepts the DAG structure returned by `WolframModel.multiway_explore_dag/2`:

      %{
        root:  canonical_key,
        nodes: %{canonical_key => %WolframModel{}},
        edges: MapSet.t({from_key, to_key})
      }

  Nodes are drawn as rounded boxes labelled with vertex count, edge count, and
  generation. Each depth level is a horizontal row. Edges are drawn as
  curved arrows. The root node is highlighted.

  ## Example

      WolframModel.multiway_explore_dag(universe, 3)
      |> WolframModel.MultiwayGraphSVG.to_svg()
      |> then(&File.write!("multiway.svg", &1))
  """

  alias Hypergraph

  @node_w 72
  @node_h 40
  @level_gap 90
  @col_gap 90

  # Pastel row colours, cycling by depth level
  @level_colors ~w(#dbeafe #dcfce7 #fef9c3 #fce7f3 #ede9fe #ffedd5 #d1fae5 #fee2e2)

  # -------------------------------------------------------------------------
  # Public API
  # -------------------------------------------------------------------------

  @doc """
  Returns an SVG string for `dag`.

  Options:
  - `:width` — canvas width in pixels. When provided the layout is scaled
    horizontally so the graph fills exactly this width. Computed automatically
    when omitted.
  - `:height` — canvas height in pixels. Computed automatically when omitted.
  """
  @spec to_svg(%{root: term(), nodes: map(), edges: MapSet.t()}, keyword()) :: String.t()
  def to_svg(dag, opts \\ []) do
    levels = bfs_levels(dag)
    positions = assign_positions(levels, dag.edges)

    auto_w = positions |> Map.values() |> Enum.map(&elem(&1, 0)) |> Enum.max(fn -> 0 end)
    auto_h = positions |> Map.values() |> Enum.map(&elem(&1, 1)) |> Enum.max(fn -> 0 end)

    content_w = auto_w + @node_w + 40
    content_h = auto_h + @node_h + 40

    w = Keyword.get(opts, :width, trunc(content_w))
    h = Keyword.get(opts, :height, trunc(content_h))

    positions =
      case Keyword.fetch(opts, :width) do
        {:ok, target_w} when content_w > 0 ->
          scale = target_w / content_w
          Map.new(positions, fn {k, {x, y}} -> {k, {Float.round(x * scale, 1), y}} end)

        _ ->
          positions
      end

    render(dag, positions, levels, w, h)
  end

  # -------------------------------------------------------------------------
  # Layout
  # -------------------------------------------------------------------------

  # BFS from root assigns depth level to every node.
  defp bfs_levels(%{root: root, edges: edges}) do
    do_bfs(MapSet.to_list(edges), [{root, 0}], %{root => 0})
  end

  defp do_bfs(_edges, [], levels), do: levels

  defp do_bfs(edges, [{key, level} | rest], levels) do
    children =
      edges
      |> Enum.filter(fn {from, _} -> from == key end)
      |> Enum.map(fn {_, to} -> to end)
      |> Enum.reject(&Map.has_key?(levels, &1))

    new_levels = Enum.reduce(children, levels, &Map.put(&2, &1, level + 1))
    do_bfs(edges, rest ++ Enum.map(children, &{&1, level + 1}), new_levels)
  end

  # Assign pixel (x, y) to each node.
  # Nodes at the same level are spread evenly; each level is a horizontal row.
  defp assign_positions(levels, edges) do
    by_level =
      levels
      |> Enum.group_by(fn {_, l} -> l end, fn {k, _} -> k end)
      |> Enum.map(fn {lvl, keys} -> {lvl, sort_by_connectivity(keys, edges)} end)
      |> Map.new()

    max_cols = by_level |> Map.values() |> Enum.map(&length/1) |> Enum.max(fn -> 1 end)

    by_level
    |> Enum.flat_map(fn {level, keys} ->
      n = length(keys)
      total_row_w = max_cols * @col_gap
      start_x = (total_row_w - (n - 1) * @col_gap) / 2 + 20

      keys
      |> Enum.with_index()
      |> Enum.map(fn {key, i} ->
        x = start_x + i * @col_gap
        y = 30.0 + level * @level_gap
        {key, {Float.round(x, 1), Float.round(y, 1)}}
      end)
    end)
    |> Map.new()
  end

  # Sort keys within a level so that nodes connected to the same parent are
  # adjacent (reduces edge crossings).
  defp sort_by_connectivity(keys, edges) do
    edge_list = MapSet.to_list(edges)

    Enum.sort_by(keys, fn key ->
      parents =
        edge_list
        |> Enum.filter(fn {_, to} -> to == key end)
        |> Enum.map(fn {from, _} -> from end)

      {-length(parents), inspect(key)}
    end)
  end

  # -------------------------------------------------------------------------
  # Rendering
  # -------------------------------------------------------------------------

  defp render(dag, positions, levels, w, h) do
    edges_svg = render_edges(dag.edges, positions)
    nodes_svg = render_nodes(dag.nodes, dag.root, positions, levels)

    """
    <svg xmlns="http://www.w3.org/2000/svg"
         width="#{w}" height="#{h}" viewBox="0 0 #{w} #{h}"
         font-family="monospace">
      <defs>
        <marker id="mw-arrow" viewBox="0 0 8 8" refX="8" refY="4"
                markerWidth="5" markerHeight="5" orient="auto-start-reverse">
          <path d="M 0 0 L 8 4 L 0 8 z" fill="#999"/>
        </marker>
      </defs>
      <rect width="#{w}" height="#{h}" fill="#f8f8f8" rx="4" stroke="#ddd"/>
      #{edges_svg}
      #{nodes_svg}
    </svg>
    """
  end

  defp render_edges(edges, positions) do
    edges
    |> MapSet.to_list()
    |> Enum.map(fn {from, to} ->
      case {positions[from], positions[to]} do
        {{x1, y1}, {x2, y2}} ->
          # Cubic bezier for a curved arrow
          mx = (x1 + x2) / 2
          cy1 = y1 + @level_gap * 0.4
          cy2 = y2 - @level_gap * 0.4

          """
          <path d="M#{x1},#{y1 + @node_h / 2} C#{mx},#{cy1} #{mx},#{cy2} #{x2},#{y2 - @node_h / 2}"
                fill="none" stroke="#bbb" stroke-width="1.5"
                marker-end="url(#mw-arrow)"/>
          """

        _ ->
          ""
      end
    end)
    |> Enum.join()
  end

  defp render_nodes(nodes, root, positions, levels) do
    positions
    |> Enum.map(fn {key, {cx, cy}} ->
      model = nodes[key]
      level = levels[key] || 0
      is_root = key == root
      render_node(model, cx, cy, level, is_root)
    end)
    |> Enum.join()
  end

  defp render_node(model, cx, cy, level, is_root) do
    {vc, ec, gen} =
      if model do
        hes = Hypergraph.hyperedges(model.hypergraph)
        vc = hes |> Enum.flat_map(& &1) |> Enum.uniq() |> length()
        {vc, length(hes), model.generation}
      else
        {0, 0, 0}
      end

    fill = Enum.at(@level_colors, rem(level, length(@level_colors)))
    stroke = if is_root, do: "#2563eb", else: "#9ca3af"
    stroke_w = if is_root, do: "2.5", else: "1"
    x = cx - @node_w / 2
    y = cy - @node_h / 2

    """
    <rect x="#{x}" y="#{y}" width="#{@node_w}" height="#{@node_h}"
          rx="7" fill="#{fill}" stroke="#{stroke}" stroke-width="#{stroke_w}"/>
    <text x="#{cx}" y="#{cy - 5}" font-size="9" text-anchor="middle" fill="#374151" font-weight="bold">V:#{vc} E:#{ec}</text>
    <text x="#{cx}" y="#{cy + 8}" font-size="8" text-anchor="middle" fill="#6b7280">gen #{gen}</text>
    """
  end
end
