defmodule WolframModel.MixProject do
  use Mix.Project

  def project do
    [
      app: :wolfram_model,
      version: "1.1.0",
      elixir: "~> 1.17",
      start_permanent: Mix.env() == :prod,
      description: description(),
      package: package(),
      deps: deps(),
      name: "WolframModel",
      source_url: "https://github.com/sragli/wolfram_model",
      docs: docs()
    ]
  end

  def application do
    [
      extra_applications: [:logger]
    ]
  end

  defp description() do
    "Elixir implementation of the Wolfram Model: hypergraph rewriting, N-pattern matching, causal networks, multiway DAG evolution, and emergent structure analysis."
  end

  defp package() do
    [
      files: ~w(lib .formatter.exs mix.exs README.md LICENSE CHANGELOG.md),
      licenses: ["Apache-2.0"],
      links: %{"GitHub" => "https://github.com/sragli/wolfram_model"}
    ]
  end

  defp docs() do
    [
      main: "WolframModel",
      extras: ["README.md", "LICENSE", "examples.livemd"]
    ]
  end

  defp deps do
    [
      {:hypergraph, git: "https://github.com/sragli/hypergraph.git"},
      {:dialyxir, "~> 1.2", only: [:dev], runtime: false},
      {:stream_data, "~> 0.6", only: :test}
    ]
  end
end
