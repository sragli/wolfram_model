defmodule WolframModel.BranchialGraphSVG do
  @moduledoc """
  SVG rendering of the branchial graph returned by `WolframModel.branchial_graph/1`.

  Each node represents a possible rule match at the current state. An edge
  connects two nodes when their matched hyperedges overlap, meaning the two
  rule applications conflict (they cannot both be applied without first
  resolving the branch).

  Nodes are placed on a circle and coloured by rule name. The label shows the
  rule name (truncated) and which hyperedges it matched.

  ## Example

      universe
      |> WolframModel.branchial_graph()
      |> WolframModel.BranchialGraphSVG.to_svg()
      |> then(&File.write!("branchial.svg", &1))
  """

  @node_r 20
  # Colour palette, cycling by rule name hash
  @colors ~w(#3b82f6 #10b981 #f59e0b #8b5cf6 #ef4444 #06b6d4 #f97316 #84cc16)

  # -------------------------------------------------------------------------
  # Public API
  # -------------------------------------------------------------------------

  @doc """
  Returns an SVG string for `branchial_data`.

  Options:
  - `:width` / `:height` — canvas size in pixels (default `480`).
  - `:title` — optional label at the top.
  """
  @spec to_svg(%{nodes: [map()], edges: [map()]}, keyword()) :: String.t()
  def to_svg(%{nodes: nodes, edges: edges}, opts \\ []) do
    w = Keyword.get(opts, :width, 480)
    h = Keyword.get(opts, :height, 480)
    title = Keyword.get(opts, :title)

    positions = circular_layout(nodes, w, h)
    render(nodes, edges, positions, w, h, title)
  end

  # -------------------------------------------------------------------------
  # Layout
  # -------------------------------------------------------------------------

  defp circular_layout(nodes, w, h) do
    n = length(nodes)
    cx = w / 2.0
    cy = h / 2.0
    r = min(w, h) / 2.0 - @node_r - 30

    nodes
    |> Enum.with_index()
    |> Map.new(fn {%{id: id}, i} ->
      a = 2 * :math.pi() * i / max(n, 1) - :math.pi() / 2
      {id, {Float.round(cx + r * :math.cos(a), 1), Float.round(cy + r * :math.sin(a), 1)}}
    end)
  end

  # -------------------------------------------------------------------------
  # Rendering
  # -------------------------------------------------------------------------

  defp render(nodes, edges, positions, w, h, title) do
    title_svg =
      if title,
        do:
          ~s|<text x="#{w / 2}" y="18" font-size="12" text-anchor="middle" fill="#444">#{title}</text>|,
        else: ""

    edge_svg = Enum.map(edges, &render_edge(&1, positions))
    node_svg = Enum.map(nodes, &render_node(&1, positions))

    # Legend: unique rule names
    rule_names = nodes |> Enum.map(& &1.rule_name) |> Enum.uniq()
    legend_svg = render_legend(rule_names, w)

    """
    <svg xmlns="http://www.w3.org/2000/svg"
         width="#{w}" height="#{h}" viewBox="0 0 #{w} #{h}"
         font-family="monospace">
      <rect width="#{w}" height="#{h}" fill="#fafafa" rx="4" stroke="#ddd"/>
      #{title_svg}
      #{Enum.join(edge_svg)}
      #{Enum.join(node_svg)}
      #{legend_svg}
    </svg>
    """
  end

  defp render_edge(%{source: s, target: t}, positions) do
    case {positions[s], positions[t]} do
      {{x1, y1}, {x2, y2}} ->
        ~s|<line x1="#{x1}" y1="#{y1}" x2="#{x2}" y2="#{y2}" stroke="#d1d5db" stroke-width="2" stroke-dasharray="5,3"/>|

      _ ->
        ""
    end
  end

  defp render_node(%{id: id, rule_name: name, matched_hyperedges: matched}, positions) do
    case positions[id] do
      nil ->
        ""

      {x, y} ->
        color = rule_color(name)
        short = name |> to_string() |> String.slice(0, 8)

        edge_summary =
          matched |> Enum.map(&edge_label/1) |> Enum.join(", ") |> String.slice(0, 14)

        """
        <circle cx="#{x}" cy="#{y}" r="#{@node_r}" fill="#{color}" fill-opacity="0.85" stroke="white" stroke-width="2"/>
        <text x="#{x}" y="#{y - 4}" font-size="8" text-anchor="middle" fill="white" font-weight="bold">#{short}</text>
        <text x="#{x}" y="#{y + 8}" font-size="7" text-anchor="middle" fill="white">#{edge_summary}</text>
        """
    end
  end

  defp render_legend(rule_names, w) do
    items =
      rule_names
      |> Enum.with_index()
      |> Enum.map(fn {name, i} ->
        color = rule_color(name)
        short = name |> to_string() |> String.slice(0, 14)
        lx = 8
        ly = 8 + i * 16

        """
        <rect x="#{lx}" y="#{ly}" width="10" height="10" rx="2" fill="#{color}"/>
        <text x="#{lx + 14}" y="#{ly + 9}" font-size="9" fill="#555">#{short}</text>
        """
      end)

    legend_h = length(rule_names) * 16 + 8
    bg_w = 120

    """
    <rect x="4" y="4" width="#{bg_w}" height="#{legend_h}" rx="4" fill="white" fill-opacity="0.8" stroke="#e5e7eb"/>
    #{Enum.join(items)}
    """
    |> then(fn legend ->
      # Only render if there are rule names and they fit
      if rule_names == [] or legend_h > w / 2, do: "", else: legend
    end)
  end

  defp edge_label(he), do: "[#{Enum.map(he, &vlabel/1) |> Enum.join(",")}]"

  defp vlabel(v) when is_integer(v), do: to_string(v)
  defp vlabel(v) when is_atom(v), do: Atom.to_string(v) |> String.slice(0, 3)
  defp vlabel({_, _, id}), do: to_string(rem(id, 100))
  defp vlabel(_), do: "?"

  # Stable colour based on rule name string hash
  defp rule_color(name) do
    idx = :erlang.phash2(name, length(@colors))
    Enum.at(@colors, idx)
  end
end
