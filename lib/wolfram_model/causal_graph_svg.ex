defmodule WolframModel.CausalGraphSVG do
  @moduledoc """
  Converts causal network data into an SVG visualization.
  """

  @node_radius 10
  @x_spacing 80
  @y_spacing 80
  @margin 40

  def to_svg(%{nodes: nodes, edges: edges}) do
    positions = layout_nodes(nodes)

    max_gen = nodes |> Enum.map(& &1.generation) |> Enum.max(fn -> 1 end)
    width = Enum.count(nodes) * @x_spacing + 2 * @margin
    height = (max_gen + 1) * @y_spacing + 2 * @margin

    edges_svg =
      edges
      |> Enum.map(fn %{source: a, target: b} ->
        {x1, y1} = positions[a]
        {x2, y2} = positions[b]

        # Retract the endpoint to the circle's edge so the arrowhead
        # is not hidden beneath the target node.
        dx = x2 - x1
        dy = y2 - y1
        len = :math.sqrt(dx * dx + dy * dy)

        {x2e, y2e} =
          if len > 0 do
            {x2 - dx / len * @node_radius, y2 - dy / len * @node_radius}
          else
            {x2, y2}
          end

        """
        <line x1="#{x1}" y1="#{y1}" x2="#{x2e}" y2="#{y2e}"
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
    # Group by generation: nodes in the same generation share a y-coordinate
    # and are spread evenly along x within that row.
    nodes
    |> Enum.group_by(& &1.generation)
    |> Enum.flat_map(fn {gen, gen_nodes} ->
      gen_nodes
      |> Enum.with_index()
      |> Enum.map(fn {%{id: id}, i} ->
        x = @margin + i * @x_spacing
        y = @margin + gen * @y_spacing
        {id, {x, y}}
      end)
    end)
    |> Map.new()
  end
end
