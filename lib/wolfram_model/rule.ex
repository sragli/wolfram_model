defmodule WolframModel.Rule do
  @moduledoc """
  Parser and printer for the canonical Wolfram rule notation.

  Rules are written in the format used throughout the Wolfram Physics Project:

      {{x,y,z},{x,w}} -> {{y,w,z},{y,z},{x,y,w}}

  Vertices are either integers (`1`, `2`, …) or symbolic names (`x`, `y`, `z`).
  The notation maps directly to the `%{pattern: ..., replacement: ..., name: ...}`
  maps consumed by `WolframModel`.

  ## Examples

      iex> WolframModel.Rule.parse("{{1,2},{1,3}} -> {{2,3},{1,4}}")
      %{pattern: [[1, 2], [1, 3]], replacement: [[2, 3], [1, 4]], name: "parsed"}

      iex> rule = %{pattern: [[1,2],[2,3]], replacement: [[1,3]], name: "join"}
      iex> WolframModel.Rule.to_string(rule)
      "{{1,2},{2,3}} -> {{1,3}}"
  """

  @type rule :: WolframModel.rule()

  @doc """
  Parses a rule string in the Wolfram notation `"LHS -> RHS"` and returns a
  rule map with `:pattern`, `:replacement`, and `:name` set to `"parsed"`.

  Vertices can be integers or symbolic names. Symbolic names become atoms.

  Raises `ArgumentError` when the string cannot be parsed.
  """
  @spec parse(String.t(), String.t()) :: rule()
  def parse(string, name \\ "parsed") do
    case String.split(string, "->", parts: 2) do
      [lhs, rhs] ->
        %{
          pattern: parse_hyperedge_list(String.trim(lhs)),
          replacement: parse_hyperedge_list(String.trim(rhs)),
          name: name
        }

      _ ->
        raise ArgumentError, "expected \"LHS -> RHS\", got: #{inspect(string)}"
    end
  end

  @doc """
  Formats a rule as a Wolfram-notation string.

      %{pattern: [[1,2],[2,3]], replacement: [[1,3]]} |> WolframModel.Rule.to_string()
      #=> "{{1,2},{2,3}} -> {{1,3}}"
  """
  @spec to_string(rule()) :: String.t()
  def to_string(rule) do
    "#{format_list(rule.pattern)} -> #{format_list(rule.replacement)}"
  end

  # --- private ---

  defp parse_hyperedge_list(string) do
    # Expect outer braces: {{...},{...}}
    trimmed = String.trim(string)

    unless String.starts_with?(trimmed, "{") and String.ends_with?(trimmed, "}") do
      raise ArgumentError, "expected hyperedge list in {{...},...} form, got: #{inspect(trimmed)}"
    end

    inner = String.slice(trimmed, 1..-2//1)

    inner
    |> split_top_level(inner, [], [], 0)
    |> Enum.map(&parse_hyperedge/1)
  end

  # Splits comma-separated hyperedge strings at the top brace level.
  # E.g. "{1,2},{2,3}" -> ["{1,2}", "{2,3}"]
  defp split_top_level(_original, "", current, acc, _depth) do
    token = current |> Enum.reverse() |> IO.iodata_to_binary() |> String.trim()

    if token == "" do
      Enum.reverse(acc)
    else
      Enum.reverse([token | acc])
    end
  end

  defp split_top_level(original, rest, current, acc, depth) do
    {ch, tail} = String.split_at(rest, 1)

    case {ch, depth} do
      {"{", _} ->
        split_top_level(original, tail, [ch | current], acc, depth + 1)

      {"}", 1} ->
        # Closing brace of a top-level hyperedge — emit token including the brace.
        token =
          [ch | current]
          |> Enum.reverse()
          |> IO.iodata_to_binary()
          |> String.trim()

        split_top_level(original, tail, [], [token | acc], 0)

      {"}", _} ->
        split_top_level(original, tail, [ch | current], acc, depth - 1)

      {",", 0} ->
        # Top-level comma between hyperedges — skip it.
        split_top_level(original, tail, current, acc, 0)

      _ ->
        split_top_level(original, tail, [ch | current], acc, depth)
    end
  end

  defp parse_hyperedge(string) do
    trimmed = String.trim(string)

    unless String.starts_with?(trimmed, "{") and String.ends_with?(trimmed, "}") do
      raise ArgumentError, "expected hyperedge in {...} form, got: #{inspect(trimmed)}"
    end

    trimmed
    |> String.slice(1..-2//1)
    |> String.split(",")
    |> Enum.map(fn token ->
      t = String.trim(token)

      case Integer.parse(t) do
        {n, ""} -> n
        _ -> String.to_atom(t)
      end
    end)
  end

  defp format_list(list_of_lists) do
    inner =
      list_of_lists
      |> Enum.map(fn he ->
        "{#{he |> Enum.map(&vertex_to_string/1) |> Enum.join(",")}}"
      end)
      |> Enum.join(",")

    "{#{inner}}"
  end

  defp vertex_to_string(v) when is_integer(v), do: Integer.to_string(v)
  defp vertex_to_string(v) when is_atom(v), do: Atom.to_string(v)
  defp vertex_to_string({a, g, id}), do: "#{a}_#{g}_#{id}"
  defp vertex_to_string(v), do: inspect(v)
end
