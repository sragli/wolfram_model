defmodule WolframModel.FoliationsTest do
  use ExUnit.Case, async: true
  alias WolframModel
  alias Hypergraph

  defp two_step_model do
    hg = Hypergraph.new() |> Hypergraph.add_hyperedge([1, 2]) |> Hypergraph.add_hyperedge([2, 3])
    rule = %{pattern: [MapSet.new([:a, :b])], replacement: [MapSet.new([:a, :new])], name: "r"}

    WolframModel.new(hg, [rule])
    |> WolframModel.evolve_step()
    |> WolframModel.evolve_step()
  end

  test "foliations returns non-empty list after evolution" do
    model = two_step_model()
    layers = WolframModel.foliations(model)
    assert is_list(layers)
    assert length(layers) >= 1
  end

  test "foliations returns empty list for unevolved model" do
    hg = Hypergraph.new() |> Hypergraph.add_hyperedge([1, 2])
    model = WolframModel.new(hg, [])
    assert WolframModel.foliations(model) == []
  end

  test "each foliation layer contains only Event structs" do
    model = two_step_model()

    WolframModel.foliations(model)
    |> List.flatten()
    |> Enum.each(fn e -> assert is_struct(e, WolframModel.Event) end)
  end

  test "every event appears in exactly one layer" do
    model = two_step_model()
    layers = WolframModel.foliations(model)
    all_ids = layers |> List.flatten() |> Enum.map(& &1.id)
    assert length(all_ids) == map_size(model.event_map)
    assert Enum.uniq(all_ids) == all_ids
  end

  test "layer 0 events have no parents" do
    model = two_step_model()
    [first_layer | _] = WolframModel.foliations(model)
    Enum.each(first_layer, fn e -> assert e.parent_ids == [] end)
  end
end
