defmodule WolframModel.HypergraphSVG do
  @moduledoc """
  SVG rendering of a `Hypergraph` state.

  Vertices are placed with a force-directed spring layout (Fruchterman-Reingold
  style). Rendering conventions:
  - **Unary** hyperedges: dashed ring around the vertex.
  - **Binary** hyperedges: directed line with an arrowhead (order preserved).
  - **N-ary** (3+) hyperedges: translucent filled polygon in a distinct colour
    per hyperedge, with a solid border.

  ## Example

      hg
      |> WolframModel.HypergraphSVG.to_svg(title: "Step 5")
      |> then(&File.write!("hypergraph.svg", &1))

      # Render whole evolution history as a horizontal strip
      model
      |> WolframModel.HypergraphSVG.evolution_to_svg(max_snapshots: 8)
      |> then(&File.write!("evolution.svg", &1))
  """

  alias Hypergraph

  @vertex_r 12
  @padding 48
  @spring_k 1.0
  @repulse_k 600.0
  @iterations 60

  # Per-arity colours for N-ary hyperedge polygons (cycling)
  @poly_colors ~w(#e74c3c #27ae60 #f39c12 #8e44ad #16a085 #2980b9 #d35400 #c0392b)

  # -------------------------------------------------------------------------
  # Public API
  # -------------------------------------------------------------------------

  @doc """
  Returns an SVG string rendering `hg`.

  Options:
  - `:width` / `:height` — canvas size in pixels (default `500`).
  - `:title` — optional label drawn at the top of the canvas.
  - `:directed` — draw arrowheads on binary edges (default `true`).
  """
  @spec to_svg(Hypergraph.t(), keyword()) :: String.t()
  def to_svg(hg, opts \\ []) do
    w = Keyword.get(opts, :width, 500)
    h = Keyword.get(opts, :height, 500)
    title = Keyword.get(opts, :title)
    directed = Keyword.get(opts, :directed, true)

    hyperedges = Hypergraph.hyperedges(hg)
    vertices = hyperedges |> Enum.flat_map(& &1) |> Enum.uniq() |> Enum.sort_by(&vsort/1)
    positions = layout(vertices, hyperedges, w, h)

    render(vertices, hyperedges, positions, w, h, directed, title, "_svg")
  end

  @doc """
  Renders the full `evolution_history` of `model` as a horizontal strip of
  panels, oldest first. Each panel shows one hypergraph snapshot.

  Options:
  - `:max_snapshots` — maximum panels to show (default `6`).
  - `:panel_size` — pixel size of each square panel (default `200`).
  """
  @spec evolution_to_svg(WolframModel.t(), keyword()) :: String.t()
  def evolution_to_svg(model, opts \\ []) do
    max_s = Keyword.get(opts, :max_snapshots, 6)
    ps = Keyword.get(opts, :panel_size, 200)

    snapshots =
      model.evolution_history
      |> Enum.reverse()
      |> Enum.take(max_s)
      |> Enum.with_index()

    n = length(snapshots)
    total_w = n * ps
    total_h = ps

    panels =
      Enum.map(snapshots, fn {hg, i} ->
        inner = panel_svg(hg, ps, i)
        ~s|<g transform="translate(#{i * ps}, 0)">#{inner}</g>|
      end)

    """
    <svg xmlns="http://www.w3.org/2000/svg"
         width="#{total_w}" height="#{total_h}"
         viewBox="0 0 #{total_w} #{total_h}"
         font-family="monospace">
      <rect width="#{total_w}" height="#{total_h}" fill="#f0f0f0"/>
      #{Enum.join(panels, "\n")}
    </svg>
    """
  end

  # -------------------------------------------------------------------------
  # Layout
  # -------------------------------------------------------------------------

  defp layout([], _hyperedges, _w, _h), do: %{}

  defp layout([v], _hyperedges, w, h) do
    %{v => {w / 2.0, h / 2.0}}
  end

  defp layout(vertices, hyperedges, w, h) do
    edge_pairs = to_pairs(hyperedges)

    vertices
    |> circle_init(w, h)
    |> spring(vertices, edge_pairs, @iterations)
    |> normalize(vertices, w, h)
  end

  defp circle_init(vertices, w, h) do
    n = length(vertices)
    cx = w / 2.0
    cy = h / 2.0
    r = min(w, h) * 0.35

    vertices
    |> Enum.with_index()
    |> Map.new(fn {v, i} ->
      a = 2 * :math.pi() * i / max(n, 1) - :math.pi() / 2
      {v, {cx + r * :math.cos(a), cy + r * :math.sin(a)}}
    end)
  end

  defp to_pairs(hyperedges) do
    hyperedges
    |> Enum.flat_map(fn
      [_] ->
        []

      [a, b] ->
        [{a, b}]

      he ->
        for i <- 0..(length(he) - 2),
            j <- (i + 1)..(length(he) - 1),
            do: {Enum.at(he, i), Enum.at(he, j)}
    end)
    |> Enum.uniq()
  end

  defp spring(positions, vertices, edge_pairs, iters) do
    Enum.reduce(1..max(iters, 1), positions, fn step, pos ->
      cool = 1.0 - step / (iters + 1)
      step_forces(pos, vertices, edge_pairs, cool)
    end)
  end

  defp step_forces(pos, vertices, edge_pairs, cooling) do
    # Repulsion: all pairs
    repulsion =
      for v1 <- vertices, v2 <- vertices, v1 != v2, reduce: %{} do
        acc ->
          {x1, y1} = pos[v1]
          {x2, y2} = pos[v2]
          dx = x1 - x2
          dy = y1 - y2
          d = max(:math.sqrt(dx * dx + dy * dy), 0.5)
          f = @repulse_k / (d * d)
          {fx, fy} = Map.get(acc, v1, {0.0, 0.0})
          Map.put(acc, v1, {fx + f * dx / d, fy + f * dy / d})
      end

    # Attraction: along edges
    attraction =
      Enum.reduce(edge_pairs, %{}, fn {v1, v2}, acc ->
        case {pos[v1], pos[v2]} do
          {nil, _} ->
            acc

          {_, nil} ->
            acc

          {{x1, y1}, {x2, y2}} ->
            dx = x2 - x1
            dy = y2 - y1
            d = max(:math.sqrt(dx * dx + dy * dy), 0.5)
            f = @spring_k * d
            {fx1, fy1} = Map.get(acc, v1, {0.0, 0.0})
            {fx2, fy2} = Map.get(acc, v2, {0.0, 0.0})

            acc
            |> Map.put(v1, {fx1 + f * dx / d, fy1 + f * dy / d})
            |> Map.put(v2, {fx2 - f * dx / d, fy2 - f * dy / d})
        end
      end)

    max_move = 20.0 * cooling

    Map.new(vertices, fn v ->
      {x, y} = pos[v]
      {rx, ry} = Map.get(repulsion, v, {0.0, 0.0})
      {ax, ay} = Map.get(attraction, v, {0.0, 0.0})
      fx = (rx + ax) * cooling * 4.0
      fy = (ry + ay) * cooling * 4.0
      mag = max(:math.sqrt(fx * fx + fy * fy), 1.0e-9)
      scale = if mag > max_move, do: max_move / mag, else: 1.0
      {v, {x + fx * scale, y + fy * scale}}
    end)
  end

  defp normalize(positions, vertices, w, h) do
    pad = @padding + @vertex_r
    xs = Enum.map(vertices, fn v -> elem(positions[v], 0) end)
    ys = Enum.map(vertices, fn v -> elem(positions[v], 1) end)
    min_x = Enum.min(xs)
    max_x = Enum.max(xs)
    min_y = Enum.min(ys)
    max_y = Enum.max(ys)
    span_x = max(max_x - min_x, 1.0)
    span_y = max(max_y - min_y, 1.0)
    scale = min((w - 2 * pad) / span_x, (h - 2 * pad) / span_y)

    Map.new(positions, fn {v, {x, y}} ->
      {v, {Float.round(pad + (x - min_x) * scale, 1), Float.round(pad + (y - min_y) * scale, 1)}}
    end)
  end

  # -------------------------------------------------------------------------
  # Rendering
  # -------------------------------------------------------------------------

  defp render(vertices, hyperedges, positions, w, h, directed, title, id_ns) do
    polys = Enum.filter(hyperedges, &(length(&1) >= 3))
    binary = Enum.filter(hyperedges, &(length(&1) == 2))
    unary = Enum.filter(hyperedges, &(length(&1) == 1))

    title_svg =
      if title,
        do:
          ~s|<text x="#{w / 2}" y="18" font-size="12" text-anchor="middle" fill="#444">#{title}</text>|,
        else: ""

    defs =
      if directed do
        """
        <defs>
          <marker id="hg-arrow#{id_ns}" viewBox="0 0 8 8" refX="8" refY="4"
                  markerWidth="5" markerHeight="5" orient="auto-start-reverse">
            <path d="M 0 0 L 8 4 L 0 8 z" fill="#555"/>
          </marker>
        </defs>
        """
      else
        ""
      end

    arrow_ref = if directed, do: ~s| marker-end="url(#hg-arrow#{id_ns})"|, else: ""

    poly_svg =
      polys
      |> Enum.with_index()
      |> Enum.map(fn {he, i} -> render_poly(he, positions, i) end)

    line_svg = Enum.map(binary, &render_line(&1, positions, arrow_ref))
    unary_svg = Enum.map(unary, &render_unary(&1, positions))
    vtx_svg = Enum.map(vertices, &render_vertex(&1, positions))

    """
    <svg xmlns="http://www.w3.org/2000/svg"
         width="#{w}" height="#{h}" viewBox="0 0 #{w} #{h}"
         font-family="monospace">
      #{defs}
      <rect width="#{w}" height="#{h}" fill="#fafafa" rx="4" stroke="#ddd"/>
      #{title_svg}
      #{Enum.join(poly_svg)}
      #{Enum.join(unary_svg)}
      #{Enum.join(line_svg)}
      #{Enum.join(vtx_svg)}
    </svg>
    """
  end

  defp panel_svg(hg, ps, idx) do
    hyperedges = Hypergraph.hyperedges(hg)
    vertices = hyperedges |> Enum.flat_map(& &1) |> Enum.uniq() |> Enum.sort_by(&vsort/1)
    positions = layout(vertices, hyperedges, ps, ps - 20)
    vc = length(vertices)
    ec = length(hyperedges)
    title = "V:#{vc} E:#{ec}"
    render(vertices, hyperedges, positions, ps, ps, true, title, "_p#{idx}")
  end

  defp render_poly(he, positions, idx) do
    color = Enum.at(@poly_colors, rem(idx, length(@poly_colors)))

    pts =
      he
      |> Enum.map(&positions[&1])
      |> Enum.reject(&is_nil/1)
      |> Enum.map(fn {x, y} -> "#{x},#{y}" end)
      |> Enum.join(" ")

    ~s|<polygon points="#{pts}" fill="#{color}" fill-opacity="0.18" stroke="#{color}" stroke-width="1.5" stroke-linejoin="round"/>|
  end

  defp render_line([v1, v2], positions, arrow_ref) do
    case {positions[v1], positions[v2]} do
      {nil, _} ->
        ""

      {_, nil} ->
        ""

      {{x, y}, {x, y}} ->
        # Self-loop
        ~s|<circle cx="#{x + @vertex_r + 6}" cy="#{y - @vertex_r - 6}" r="10" fill="none" stroke="#888" stroke-width="1.5"/>|

      {{x1, y1}, {x2, y2}} ->
        dx = x2 - x1
        dy = y2 - y1
        d = max(:math.sqrt(dx * dx + dy * dy), 1.0)
        # Shorten endpoint to sit at the circle edge
        ratio = (d - @vertex_r - 1) / d
        ex = Float.round(x1 + dx * ratio, 1)
        ey = Float.round(y1 + dy * ratio, 1)

        ~s|<line x1="#{x1}" y1="#{y1}" x2="#{ex}" y2="#{ey}" stroke="#555" stroke-width="1.8"#{arrow_ref}/>|
    end
  end

  defp render_unary([v], positions) do
    case positions[v] do
      nil ->
        ""

      {x, y} ->
        ~s|<circle cx="#{x}" cy="#{y}" r="#{@vertex_r + 6}" fill="none" stroke="#aaa" stroke-width="1.5" stroke-dasharray="4,3"/>|
    end
  end

  defp render_vertex(v, positions) do
    case positions[v] do
      nil ->
        ""

      {x, y} ->
        label = vlabel(v)

        """
        <circle cx="#{x}" cy="#{y}" r="#{@vertex_r}" fill="#4f83cc" stroke="#2c5f8a" stroke-width="1.5"/>
        <text x="#{x}" y="#{y + 4}" font-size="9" text-anchor="middle" fill="white" font-weight="bold">#{label}</text>
        """
    end
  end

  defp vlabel(v) when is_integer(v), do: to_string(v)
  defp vlabel(v) when is_atom(v), do: v |> Atom.to_string() |> String.slice(0, 5)
  defp vlabel({a, _g, id}), do: "#{String.first(Atom.to_string(a))}#{rem(id, 100)}"
  defp vlabel(v), do: inspect(v) |> String.slice(0, 5)

  defp vsort(v) when is_integer(v), do: {0, v, ""}
  defp vsort(v) when is_atom(v), do: {1, 0, Atom.to_string(v)}
  defp vsort({a, g, id}), do: {2, g * 1_000_000 + id, Atom.to_string(a)}
  defp vsort(_), do: {3, 0, ""}
end
