defmodule WolframModel.GeodesicPlotSVG do
  @moduledoc """
  SVG line chart of geodesic ball growth in a `Hypergraph`.

  For each seed vertex, the plot shows `V(r)` — the number of vertices
  reachable within geodesic distance `r` — as a function of `r`. In a
  *d*-dimensional space, `V(r) ~ r^d`, so the slope of the log-log version
  gives an estimate of the effective spatial dimension.

  Two panels are rendered side-by-side:
  1. **Linear** — `V(r)` vs `r` (raw ball volumes).
  2. **Log-log** — `log V(r)` vs `log r` with the best-fit line whose slope
     is the estimated dimension `d`.

  ## Example

      evolved.hypergraph
      |> WolframModel.GeodesicPlotSVG.to_svg(seeds: 5, title: "Dimension estimate")
      |> then(&File.write!("geodesic.svg", &1))
  """

  alias Hypergraph

  @panel_w 340
  @panel_h 280
  @gap 40
  @margin %{top: 40, right: 20, bottom: 50, left: 55}

  @line_colors ~w(#3b82f6 #ef4444 #10b981 #f59e0b #8b5cf6 #06b6d4 #f97316)

  # -------------------------------------------------------------------------
  # Public API
  # -------------------------------------------------------------------------

  @doc """
  Returns an SVG string containing the geodesic ball growth chart.

  Options:
  - `:seeds` — number of seed vertices to sample (default `5`).
  - `:title` — overall title shown above both panels.
  """
  @spec to_svg(Hypergraph.t(), keyword()) :: String.t()
  def to_svg(hg, opts \\ []) do
    n_seeds = Keyword.get(opts, :seeds, 5)
    title = Keyword.get(opts, :title)

    hyperedges = Hypergraph.hyperedges(hg)
    vertices = hyperedges |> Enum.flat_map(& &1) |> Enum.uniq()

    if length(vertices) < 2 do
      empty_svg(title)
    else
      seeds = Enum.take(vertices, min(n_seeds, length(vertices)))

      ball_data =
        seeds
        |> Enum.with_index()
        |> Enum.map(fn {seed, i} ->
          distances = hg_bfs(hyperedges, seed)
          max_r = distances |> Map.values() |> Enum.max(fn -> 0 end)

          series =
            for r <- 1..max(max_r, 1), do: {r, Enum.count(distances, fn {_, d} -> d <= r end)}

          {i, series}
        end)

      total_w = @panel_w * 2 + @gap + 20
      total_h = @panel_h + 20

      linear_panel = render_panel(ball_data, :linear, 10, 10)
      loglog_panel = render_loglog_panel(ball_data, @panel_w + @gap + 10, 10)

      title_svg =
        if title,
          do:
            ~s|<text x="#{total_w / 2}" y="16" font-size="13" text-anchor="middle" fill="#333" font-family="monospace">#{title}</text>|,
          else: ""

      """
      <svg xmlns="http://www.w3.org/2000/svg"
           width="#{total_w}" height="#{total_h + if title, do: 20, else: 0}"
           viewBox="0 0 #{total_w} #{total_h + if title, do: 20, else: 0}"
           font-family="monospace">
        <rect width="#{total_w}" height="#{total_h + if title, do: 20, else: 0}" fill="#fafafa" rx="4" stroke="#ddd"/>
        #{title_svg}
        #{if title, do: ~s|<g transform="translate(0,20)">#{linear_panel}#{loglog_panel}</g>|, else: linear_panel <> loglog_panel}
      </svg>
      """
    end
  end

  # -------------------------------------------------------------------------
  # Panels
  # -------------------------------------------------------------------------

  defp render_panel(ball_data, :linear, ox, oy) do
    all_r = ball_data |> Enum.flat_map(fn {_, s} -> Enum.map(s, &elem(&1, 0)) end)
    all_v = ball_data |> Enum.flat_map(fn {_, s} -> Enum.map(s, &elem(&1, 1)) end)

    max_r = Enum.max(all_r, fn -> 1 end)
    max_v = Enum.max(all_v, fn -> 1 end)

    pw = @panel_w - @margin.left - @margin.right
    ph = @panel_h - @margin.top - @margin.bottom

    sx = fn r -> @margin.left + pw * (r - 1) / max(max_r - 1, 1) end
    sy = fn v -> @margin.top + ph * (1 - v / max_v) end

    axes = axes_linear(max_r, max_v, @margin, pw, ph)
    labels = panel_labels("V(r)", "r", "Geodesic Ball Growth", @panel_w, @margin)
    lines = Enum.map(ball_data, fn {i, series} -> polyline(series, sx, sy, i) end)

    """
    <g transform="translate(#{ox}, #{oy})">
      <rect width="#{@panel_w}" height="#{@panel_h}" fill="white" rx="3" stroke="#e5e7eb"/>
      #{axes}
      #{labels}
      #{Enum.join(lines)}
    </g>
    """
  end

  defp render_loglog_panel(ball_data, ox, oy) do
    # Filter out r=1 (log 1 = 0), keep only points where V > 1
    log_data =
      ball_data
      |> Enum.map(fn {i, series} ->
        log_series =
          series
          |> Enum.filter(fn {r, v} -> r > 1 and v > 1 end)
          |> Enum.map(fn {r, v} -> {:math.log(r), :math.log(v)} end)

        {i, log_series}
      end)
      |> Enum.filter(fn {_, s} -> length(s) >= 2 end)

    all_x = log_data |> Enum.flat_map(fn {_, s} -> Enum.map(s, &elem(&1, 0)) end)
    all_y = log_data |> Enum.flat_map(fn {_, s} -> Enum.map(s, &elem(&1, 1)) end)

    {min_x, max_x} = if all_x == [], do: {0.0, 1.0}, else: {Enum.min(all_x), Enum.max(all_x)}
    {min_y, max_y} = if all_y == [], do: {0.0, 1.0}, else: {Enum.min(all_y), Enum.max(all_y)}
    span_x = max(max_x - min_x, 0.1)
    span_y = max(max_y - min_y, 0.1)

    pw = @panel_w - @margin.left - @margin.right
    ph = @panel_h - @margin.top - @margin.bottom

    sx = fn lx -> @margin.left + pw * (lx - min_x) / span_x end
    sy = fn ly -> @margin.top + ph * (1 - (ly - min_y) / span_y) end

    dim = estimate_dim(log_data)

    fit_line =
      if dim && all_x != [] do
        x0 = min_x
        x1 = max_x
        # Use mean y-intercept
        y_intercepts =
          log_data
          |> Enum.flat_map(fn {_, s} ->
            Enum.map(s, fn {lx, ly} -> ly - dim * lx end)
          end)

        b = Enum.sum(y_intercepts) / max(length(y_intercepts), 1)
        lx0 = sx.(x0)
        ly0 = sy.(b + dim * x0)
        lx1 = sx.(x1)
        ly1 = sy.(b + dim * x1)

        dim_str = Float.round(dim, 2)

        ~s|<line x1="#{lx0}" y1="#{ly0}" x2="#{lx1}" y2="#{ly1}" stroke="#374151" stroke-width="1.5" stroke-dasharray="6,3"/><text x="#{lx1 - 4}" y="#{ly1 - 6}" font-size="9" fill="#374151" text-anchor="end">d≈#{dim_str}</text>|
      else
        ""
      end

    axes = axes_loglog(min_x, max_x, min_y, max_y, @margin, pw, ph)
    labels = panel_labels("log V(r)", "log r", "Log-log (dimension fit)", @panel_w, @margin)
    lines = Enum.map(log_data, fn {i, series} -> polyline(series, sx, sy, i) end)

    """
    <g transform="translate(#{ox}, #{oy})">
      <rect width="#{@panel_w}" height="#{@panel_h}" fill="white" rx="3" stroke="#e5e7eb"/>
      #{axes}
      #{labels}
      #{fit_line}
      #{Enum.join(lines)}
    </g>
    """
  end

  # -------------------------------------------------------------------------
  # Chart primitives
  # -------------------------------------------------------------------------

  defp polyline(series, sx, sy, color_idx) do
    color = Enum.at(@line_colors, rem(color_idx, length(@line_colors)))

    pts =
      series
      |> Enum.map(fn {a, b} -> "#{Float.round(sx.(a), 1)},#{Float.round(sy.(b), 1)}" end)
      |> Enum.join(" ")

    dots =
      Enum.map(series, fn {a, b} ->
        ~s|<circle cx="#{Float.round(sx.(a), 1)}" cy="#{Float.round(sy.(b), 1)}" r="3" fill="#{color}"/>|
      end)

    """
    <polyline points="#{pts}" fill="none" stroke="#{color}" stroke-width="2" stroke-linejoin="round"/>
    #{Enum.join(dots)}
    """
  end

  defp axes_linear(max_r, max_v, m, pw, ph) do
    x_axis =
      ~s|<line x1="#{m.left}" y1="#{m.top + ph}" x2="#{m.left + pw}" y2="#{m.top + ph}" stroke="#9ca3af" stroke-width="1"/>|

    y_axis =
      ~s|<line x1="#{m.left}" y1="#{m.top}" x2="#{m.left}" y2="#{m.top + ph}" stroke="#9ca3af" stroke-width="1"/>|

    x_ticks =
      1..max(max_r, 1)
      |> Enum.filter(fn r -> max_r <= 10 or rem(r, max(div(max_r, 5), 1)) == 0 end)
      |> Enum.map(fn r ->
        x = m.left + pw * (r - 1) / max(max_r - 1, 1)

        """
        <line x1="#{x}" y1="#{m.top + ph}" x2="#{x}" y2="#{m.top + ph + 4}" stroke="#9ca3af"/>
        <text x="#{x}" y="#{m.top + ph + 14}" font-size="9" text-anchor="middle" fill="#6b7280">#{r}</text>
        """
      end)

    v_step = nice_step(max_v, 5)

    y_ticks =
      0..max_v//max(v_step, 1)
      |> Enum.map(fn v ->
        y = m.top + ph * (1 - v / max_v)

        """
        <line x1="#{m.left - 4}" y1="#{y}" x2="#{m.left}" y2="#{y}" stroke="#9ca3af"/>
        <text x="#{m.left - 6}" y="#{y + 4}" font-size="9" text-anchor="end" fill="#6b7280">#{v}</text>
        """
      end)

    x_axis <> y_axis <> Enum.join(x_ticks) <> Enum.join(y_ticks)
  end

  defp axes_loglog(min_x, max_x, min_y, max_y, m, pw, ph) do
    span_x = max(max_x - min_x, 0.1)
    span_y = max(max_y - min_y, 0.1)

    x_axis =
      ~s|<line x1="#{m.left}" y1="#{m.top + ph}" x2="#{m.left + pw}" y2="#{m.top + ph}" stroke="#9ca3af" stroke-width="1"/>|

    y_axis =
      ~s|<line x1="#{m.left}" y1="#{m.top}" x2="#{m.left}" y2="#{m.top + ph}" stroke="#9ca3af" stroke-width="1"/>|

    # X ticks: a few evenly-spaced values in log space
    x_ticks =
      Enum.map(0..4, fn i ->
        lx = min_x + span_x * i / 4
        x = m.left + pw * (lx - min_x) / span_x
        label = Float.round(:math.exp(lx), 1)

        """
        <line x1="#{x}" y1="#{m.top + ph}" x2="#{x}" y2="#{m.top + ph + 4}" stroke="#9ca3af"/>
        <text x="#{x}" y="#{m.top + ph + 14}" font-size="9" text-anchor="middle" fill="#6b7280">#{label}</text>
        """
      end)

    y_ticks =
      Enum.map(0..4, fn i ->
        ly = min_y + span_y * i / 4
        y = m.top + ph * (1 - (ly - min_y) / span_y)
        label = Float.round(:math.exp(ly), 1)

        """
        <line x1="#{m.left - 4}" y1="#{y}" x2="#{m.left}" y2="#{y}" stroke="#9ca3af"/>
        <text x="#{m.left - 6}" y="#{y + 4}" font-size="9" text-anchor="end" fill="#6b7280">#{label}</text>
        """
      end)

    x_axis <> y_axis <> Enum.join(x_ticks) <> Enum.join(y_ticks)
  end

  defp panel_labels(y_label, x_label, title, panel_w, m) do
    ph = @panel_h - m.top - m.bottom
    pw = panel_w - m.left - m.right

    ~s|<text x="#{m.left + pw / 2}" y="#{@panel_h - 8}" font-size="10" text-anchor="middle" fill="#6b7280">#{x_label}</text>| <>
      ~s|<text x="12" y="#{m.top + ph / 2}" font-size="10" text-anchor="middle" fill="#6b7280" transform="rotate(-90, 12, #{m.top + ph / 2})">#{y_label}</text>| <>
      ~s|<text x="#{m.left + pw / 2}" y="#{m.top - 10}" font-size="11" text-anchor="middle" fill="#374151" font-weight="bold">#{title}</text>|
  end

  # -------------------------------------------------------------------------
  # Helpers
  # -------------------------------------------------------------------------

  # Hypergraph BFS: one hop = traversal through an entire hyperedge.
  defp hg_bfs(hyperedges, source) do
    do_hg_bfs(hyperedges, [{source, 0}], %{source => 0})
  end

  defp do_hg_bfs(_hes, [], visited), do: visited

  defp do_hg_bfs(hes, [{v, dist} | queue], visited) do
    new_nbrs =
      hes
      |> Enum.filter(&(v in &1))
      |> Enum.flat_map(& &1)
      |> Enum.reject(&Map.has_key?(visited, &1))
      |> Enum.uniq()

    new_visited = Enum.reduce(new_nbrs, visited, &Map.put(&2, &1, dist + 1))
    do_hg_bfs(hes, queue ++ Enum.map(new_nbrs, &{&1, dist + 1}), new_visited)
  end

  defp estimate_dim(log_data) do
    points =
      Enum.flat_map(log_data, fn {_, s} ->
        Enum.map(s, fn {lx, ly} -> {lx, ly} end)
      end)

    if length(points) < 2 do
      nil
    else
      {xs, ys} = Enum.unzip(points)
      n = length(xs)
      mx = Enum.sum(xs) / n
      my = Enum.sum(ys) / n
      num = Enum.zip(xs, ys) |> Enum.reduce(0.0, fn {x, y}, a -> a + (x - mx) * (y - my) end)
      den = Enum.reduce(xs, 0.0, fn x, a -> a + (x - mx) * (x - mx) end)
      if den == 0.0, do: nil, else: num / den
    end
  end

  defp nice_step(max_val, target_ticks) do
    raw = max_val / target_ticks
    magnitude = :math.pow(10, Float.floor(:math.log10(max(raw, 1))))
    step = trunc(Float.ceil(raw / magnitude) * magnitude)
    max(step, 1)
  end

  defp empty_svg(title) do
    msg = "Not enough vertices to plot"

    title_svg =
      if title,
        do:
          ~s|<text x="200" y="20" font-size="12" text-anchor="middle" fill="#444">#{title}</text>|,
        else: ""

    """
    <svg xmlns="http://www.w3.org/2000/svg" width="400" height="60" font-family="monospace">
      <rect width="400" height="60" fill="#fafafa" rx="4" stroke="#ddd"/>
      #{title_svg}
      <text x="200" y="40" font-size="11" text-anchor="middle" fill="#9ca3af">#{msg}</text>
    </svg>
    """
  end
end
