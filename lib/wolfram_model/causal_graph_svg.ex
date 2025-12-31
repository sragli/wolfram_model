defmodule CausalGraphSVG do
  @moduledoc """
  Converts causal network data into an SVG visualization.
  """

  @node_radius 10
  @x_spacing 80
  @y_spacing 80
  @margin 40

  def to_svg(%{nodes: nodes, edges: edges}) do
    positions = layout_nodes(nodes)

    width = Enum.count(nodes) * @x_spacing + 2 * @margin
    height = @y_spacing * 3

    edges_svg =
      edges
      |> Enum.map(fn %{from: a, to: b} ->
        {x1, y1} = positions[a]
        {x2, y2} = positions[b]

        """
        <line x1="#{x1}" y1="#{y1}" x2="#{x2}" y2="#{y2}"
              stroke="black" stroke-width="1.5"
              marker-end="url(#arrow)"/>
        """
      end)

    nodes_svg =
      nodes
      |> Enum.map(fn %{id: id} ->
        {x, y} = positions[id]

        """
        <circle cx="#{x}" cy="#{y}" r="#{@node_radius}"
                fill="#4f83cc" stroke="black"/>
        <text x="#{x}" y="#{y + 4}" font-size="10"
              text-anchor="middle" fill="white">#{id}</text>
        """
      end)

    """
    <svg xmlns="http://www.w3.org/2000/svg"
         width="#{width}" height="#{height}"
         viewBox="0 0 #{width} #{height}">

      <defs>
        <marker id="arrow" viewBox="0 0 10 10"
                refX="10" refY="5"
                markerWidth="6" markerHeight="6"
                orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="black"/>
        </marker>
      </defs>

      #{Enum.join(edges_svg, "\n")}
      #{Enum.join(nodes_svg, "\n")}
    </svg>
    """
  end

  defp layout_nodes(nodes) do
    nodes
    |> Enum.with_index()
    |> Enum.map(fn {%{id: id}, i} ->
      x = @margin + i * @x_spacing
      y = @margin + @y_spacing
      {id, {x, y}}
    end)
    |> Map.new()
  end
end
