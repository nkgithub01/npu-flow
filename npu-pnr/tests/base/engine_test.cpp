#include <algorithm>
#include <ranges>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "base/common.hpp"
#include "base/engine.hpp"
#include "gtest_helpers.hpp"

using namespace base;

namespace {

class PnRNetlistFormatTestFixture : protected PnRNetlistFormat {
public:
  static auto testParseHyperEdgeKey(const std::string &key) {
    return parseHyperEdgeKey(key);
  }

  static std::string testGenerateHyperEdgeKey(const std::vector<ID> &froms,
                                              const std::vector<ID> &tos) {
    return generateHyperEdgeKey(froms, tos);
  }
};

} // namespace

TEST(PnRNetlistFormatParseHyperEdgeKeyTest, ValidSingleSourceAndTarget) {
  const auto [froms, tos] =
      PnRNetlistFormatTestFixture::testParseHyperEdgeKey("node1->node2");

  ASSERT_EQ(froms.size(), 1U);
  ASSERT_EQ(tos.size(), 1U);
  EXPECT_EQ(froms[0], "node1");
  EXPECT_EQ(tos[0], "node2");
}

TEST(PnRNetlistFormatParseHyperEdgeKeyTest,
     ValidMultipleSourcesAndSingleTarget) {
  const auto [froms, tos] =
      PnRNetlistFormatTestFixture::testParseHyperEdgeKey("node1,node2->node3");

  ASSERT_EQ(froms.size(), 2U);
  ASSERT_EQ(tos.size(), 1U);
  EXPECT_EQ(froms[0], "node1");
  EXPECT_EQ(froms[1], "node2");
  EXPECT_EQ(tos[0], "node3");
}

TEST(PnRNetlistFormatParseHyperEdgeKeyTest,
     ValidSingleSourceAndMultipleTargets) {
  const auto [froms, tos] = PnRNetlistFormatTestFixture::testParseHyperEdgeKey(
      "node1->node2,node3,node4");

  ASSERT_EQ(froms.size(), 1U);
  ASSERT_EQ(tos.size(), 3U);
  EXPECT_EQ(froms[0], "node1");
  EXPECT_EQ(tos[0], "node2");
  EXPECT_EQ(tos[1], "node3");
  EXPECT_EQ(tos[2], "node4");
}

TEST(PnRNetlistFormatParseHyperEdgeKeyTest, HandlesWhitespaceCorrectly) {
  const auto [froms, tos] = PnRNetlistFormatTestFixture::testParseHyperEdgeKey(
      " node1 , node2 -> node3 , node4 ");

  ASSERT_EQ(froms.size(), 2U);
  ASSERT_EQ(tos.size(), 2U);
  EXPECT_EQ(froms[0], "node1");
  EXPECT_EQ(froms[1], "node2");
  EXPECT_EQ(tos[0], "node3");
  EXPECT_EQ(tos[1], "node4");
}

TEST(PnRNetlistFormatParseHyperEdgeKeyTest, MissingArrowThrowsException) {
  gtest_helpers::ExpectThrowsWithSubstrings<std::runtime_error>(
      [&]() {
        (void)PnRNetlistFormatTestFixture::testParseHyperEdgeKey("node1,node2");
      },
      {"missing '->'"});
}

TEST(PnRNetlistFormatParseHyperEdgeKeyTest, EmptyFromThrowsException) {
  gtest_helpers::ExpectThrowsWithSubstrings<std::runtime_error>(
      [&]() {
        (void)PnRNetlistFormatTestFixture::testParseHyperEdgeKey("->node2");
      },
      {"empty froms or tos"});
}

TEST(PnRNetlistFormatParseHyperEdgeKeyTest, EmptyToThrowsException) {
  gtest_helpers::ExpectThrowsWithSubstrings<std::runtime_error>(
      [&]() {
        (void)PnRNetlistFormatTestFixture::testParseHyperEdgeKey("node1->");
      },
      {"empty froms or tos"});
}

TEST(PnRNetlistFormatGenerateHyperEdgeKeyTest, SingleSourceAndTarget) {
  const std::string key =
      PnRNetlistFormatTestFixture::testGenerateHyperEdgeKey({"node1"},
                                                            {"node2"});

  EXPECT_EQ(key, "node1->node2");
}

TEST(PnRNetlistFormatGenerateHyperEdgeKeyTest, MultipleSourcesAndTargets) {
  const std::string key = PnRNetlistFormatTestFixture::testGenerateHyperEdgeKey(
      {"node1", "node2"}, {"node3", "node4"});

  EXPECT_EQ(key, "node1,node2->node3,node4");
}

TEST(PnRNetlistFormatGenerateHyperEdgeKeyTest, RoundTripConsistency) {
  const std::vector<std::string> froms = {"a", "b", "c"};
  const std::vector<std::string> tos = {"x", "y"};
  const std::string key =
      PnRNetlistFormatTestFixture::testGenerateHyperEdgeKey(froms, tos);
  const auto [parsed_froms, parsed_tos] =
      PnRNetlistFormatTestFixture::testParseHyperEdgeKey(key);

  EXPECT_EQ(parsed_froms, froms);
  EXPECT_EQ(parsed_tos, tos);
}

TEST(PnRNetlistReaderFromTomlBasicParsingTest, MinimalValidNetlist) {
  const std::string toml = R"(
    [node]
    n1 = "0,0"
    n2 = "0,1"

    [edge]
    e1 = [ "n1->n2" ]

    [linking]
  )";

  const auto reader = PnRNetlistReader::fromTOML(toml);
  const auto nodes = reader.getNodes();
  const auto edges = reader.getEdges();
  const auto linkings = reader.getLinkings();

  EXPECT_EQ(nodes.size(), 2U);
  EXPECT_EQ(edges.size(), 1U);
  EXPECT_TRUE(linkings.empty());
}

TEST(PnRNetlistReaderFromTomlBasicParsingTest, EdgeWithUniformDepth) {
  const std::string toml = R"(
    [node]
    n1 = "0,0"
    n2 = "0,1"

    [edge]
    e1 = [ "n1->n2", [5], 1920 ]

    [linking]
  )";

  const auto reader = PnRNetlistReader::fromTOML(toml);
  const auto edges = reader.getEdges();

  ASSERT_EQ(edges.size(), 1U);
  ASSERT_TRUE(edges[0].depth.has_value());
  EXPECT_EQ(edges[0].depth->size(), 2U);
  EXPECT_EQ(edges[0].depth->at(0), 5U);
  EXPECT_EQ(edges[0].depth->at(1), 5U);
  EXPECT_EQ(edges[0].width.value(), 1920U);
}

TEST(PnRNetlistReaderFromTomlBasicParsingTest, EdgeWithPerNodeDepth) {
  const std::string toml = R"(
    [node]
    n1 = "0,0"
    n2 = "0,1"
    n3 = "0,2"

    [edge]
    e1 = [ "n1->n2,n3", [2, 3, 4], 1920 ]

    [linking]
  )";

  const auto reader = PnRNetlistReader::fromTOML(toml);
  const auto edges = reader.getEdges();

  ASSERT_EQ(edges.size(), 1U);
  ASSERT_TRUE(edges[0].depth.has_value());
  EXPECT_EQ(edges[0].depth->size(), 3U);
  EXPECT_EQ(edges[0].depth->at(0), 2U);
  EXPECT_EQ(edges[0].depth->at(1), 3U);
  EXPECT_EQ(edges[0].depth->at(2), 4U);
}

TEST(PnRNetlistReaderFromTomlBasicParsingTest, AutoAddNodesFromEdges) {
  const std::string toml = R"(
    [node]

    [edge]
    e1 = [ "n1->n2,n3" ]

    [linking]
  )";

  const auto reader = PnRNetlistReader::fromTOML(toml);
  const auto nodes = reader.getNodes();

  EXPECT_EQ(nodes.size(), 3U);
  for (const auto &node : nodes) {
    EXPECT_FALSE(node.position.has_value());
  }
}

TEST(PnRNetlistReaderFromTomlBasicParsingTest, SelfLoopEdge) {
  const std::string toml = R"(
    [node]
    n1 = "0,0"

    [edge]
    e1 = [ "n1->n1" ]

    [linking]
  )";

  const auto reader = PnRNetlistReader::fromTOML(toml);
  const auto edges = reader.getEdges();

  ASSERT_EQ(edges.size(), 1U);
  EXPECT_EQ(edges[0].start_node, "n1");
  EXPECT_EQ(edges[0].target_nodes[0], "n1");
}

TEST(PnRNetlistReaderFromTomlBasicParsingTest, MultipleEdgesWithSameNodes) {
  const std::string toml = R"(
    [node]
    n1 = "0,0"
    n2 = "0,1"

    [edge]
    e1 = [ "n1->n2", [2], 1920 ]
    e2 = [ "n1->n2", [3], 1920 ]
  )";

  const auto reader = PnRNetlistReader::fromTOML(toml);
  const auto edges = reader.getEdges();

  ASSERT_EQ(edges.size(), 2U);
  EXPECT_TRUE(edges[0].id == "e1" || edges[1].id == "e1");
  EXPECT_TRUE(edges[0].id == "e2" || edges[1].id == "e2");
}

TEST(PnRNetlistReaderFromTomlValidationErrorsTest,
     IsolatedNodeThrowsException) {
  const std::string toml = R"(
    [node]
    n1 = "0,0"
    n2 = "0,1"
    isolated = "1,1"

    [edge]
    e1 = [ "n1->n2" ]

    [linking]
  )";

  gtest_helpers::ExpectThrowsWithSubstrings<std::runtime_error>(
      [&]() { (void)PnRNetlistReader::fromTOML(toml); }, {"not connected"});
}

TEST(PnRNetlistReaderFromTomlValidationErrorsTest,
     EdgeWithWrongNumberOfFields) {
  const std::string toml = R"(
    [node]
    n1 = "0,0"
    n2 = "0,1"

    [edge]
    e1 = [ "n1->n2", [5] ]

    [linking]
  )";

  gtest_helpers::ExpectThrowsWithSubstrings<std::runtime_error>(
      [&]() { (void)PnRNetlistReader::fromTOML(toml); },
      {"must have exactly 1"});
}

TEST(PnRNetlistReaderFromTomlValidationErrorsTest, EdgeDepthCountMismatch) {
  const std::string toml = R"(
    [node]
    n1 = "0,0"
    n2 = "0,1"
    n3 = "0,2"

    [edge]
    e1 = [ "n1->n2,n3", [2, 3], 1920 ]

    [linking]
  )";

  gtest_helpers::ExpectThrowsWithSubstrings<std::runtime_error>(
      [&]() { (void)PnRNetlistReader::fromTOML(toml); },
      {"depth field must have exactly"});
}

TEST(PnRNetlistReaderFromTomlValidationErrorsTest, ZeroBufferWidth) {
  const std::string toml = R"(
    [node]
    n1 = "0,0"
    n2 = "0,1"

    [edge]
    e1 = [ "n1->n2", [5], 0 ]

    [linking]
  )";

  gtest_helpers::ExpectThrowsWithSubstrings<std::runtime_error>(
      [&]() { (void)PnRNetlistReader::fromTOML(toml); },
      {"zero buffer width"});
}

TEST(PnRNetlistReaderFromTomlValidationErrorsTest, ZeroBufferDepth) {
  const std::string toml = R"(
    [node]
    n1 = "0,0"
    n2 = "0,1"

    [edge]
    e1 = [ "n1->n2", [0], 1920 ]

    [linking]
  )";

  gtest_helpers::ExpectThrowsWithSubstrings<std::runtime_error>(
      [&]() { (void)PnRNetlistReader::fromTOML(toml); },
      {"zero buffer depth"});
}

TEST(PnRNetlistReaderFromTomlValidationErrorsTest,
     LinkingWithNonExistentFromEdge) {
  const std::string toml = R"(
    [node]
    n1 = "0,0"
    n2 = "0,1"

    [edge]
    e1 = [ "n1->n2" ]

    [linking]
    link1 = "nonexistent->e1"
  )";

  gtest_helpers::ExpectThrowsWithSubstrings<std::runtime_error>(
      [&]() { (void)PnRNetlistReader::fromTOML(toml); },
      {"'from' edge", "does not exist"});
}

TEST(PnRNetlistReaderFromTomlValidationErrorsTest,
     LinkingWithNonExistentToEdge) {
  const std::string toml = R"(
    [node]
    n1 = "0,0"
    n2 = "0,1"

    [edge]
    e1 = [ "n1->n2" ]

    [linking]
    link1 = "e1->nonexistent"
  )";

  gtest_helpers::ExpectThrowsWithSubstrings<std::runtime_error>(
      [&]() { (void)PnRNetlistReader::fromTOML(toml); },
      {"'to' edge", "does not exist"});
}

TEST(PnRNetlistReaderFromTomlValidationErrorsTest,
     InvalidLinkingTypeManyToMany) {
  const std::string toml = R"(
    [node]
    n1 = "0,0"
    n2 = "0,1"
    n3 = "0,2"

    [edge]
    e1 = [ "n1->n2" ]
    e2 = [ "n1->n2" ]
    e3 = [ "n2->n3" ]
    e4 = [ "n2->n3" ]

    [linking]
    link1 = "e1,e2->e3,e4"
  )";

  gtest_helpers::ExpectThrowsWithSubstrings<std::runtime_error>(
      [&]() { (void)PnRNetlistReader::fromTOML(toml); },
      {"one-to-one, one-to-many"});
}

TEST(PnRNetlistReaderFromTomlValidationErrorsTest,
     LinkingEdgesWithIncompatibleWidthsInFrom) {
  const std::string toml = R"(
    [node]
    n1 = "0,0"
    n2 = "0,1"
    n3 = "0,2"

    [edge]
    e1 = [ "n1->n2", [2], 1920 ]
    e2 = [ "n1->n2", [2], 3840 ]
    e3 = [ "n2->n3", [2], 1920 ]

    [linking]
    link1 = "e1,e2->e3"
  )";

  gtest_helpers::ExpectThrowsWithSubstrings<std::runtime_error>(
      [&]() { (void)PnRNetlistReader::fromTOML(toml); },
      {"'from' edges", "incompatible buffer demands"});
}

TEST(PnRNetlistReaderFromTomlValidationErrorsTest,
     LinkingEdgesWithIncompatibleWidthsInTo) {
  const std::string toml = R"(
    [node]
    n1 = "0,0"
    n2 = "0,1"
    n3 = "0,2"

    [edge]
    e1 = [ "n1->n2", [2], 1920 ]
    e2 = [ "n2->n3", [2], 1920 ]
    e3 = [ "n2->n3", [2], 3840 ]

    [linking]
    link1 = "e1->e2,e3"
  )";

  gtest_helpers::ExpectThrowsWithSubstrings<std::runtime_error>(
      [&]() { (void)PnRNetlistReader::fromTOML(toml); },
      {"'to' edges", "incompatible buffer demands"});
}

TEST(PnRNetlistReaderFromTomlValidationErrorsTest,
     LinkingToEdgesWithoutSharedStartNode) {
  const std::string toml = R"(
    [node]
    n1 = "0,0"
    n2 = "0,1"
    n3 = "0,2"
    n4 = "0,3"

    [edge]
    e1 = [ "n1->n2", [2], 1920 ]
    e2 = [ "n2->n3", [2], 1920 ]
    e3 = [ "n4->n3", [2], 1920 ]

    [linking]
    link1 = "e1->e2,e3"
  )";

  gtest_helpers::ExpectThrowsWithSubstrings<std::runtime_error>(
      [&]() { (void)PnRNetlistReader::fromTOML(toml); },
      {"'to' edges", "same start node"});
}

TEST(PnRNetlistReaderFromTomlValidationErrorsTest,
     LinkingFromEdgesWithoutSharedTargetNode) {
  const std::string toml = R"(
    [node]
    n1 = "0,0"
    n2 = "0,1"
    n3 = "0,2"
    n4 = "0,3"

    [edge]
    e1 = [ "n1->n2", [2], 1920 ]
    e2 = [ "n1->n3", [2], 1920 ]
    e3 = [ "n2->n4", [2], 1920 ]

    [linking]
    link1 = "e1,e2->e3"
  )";

  gtest_helpers::ExpectThrowsWithSubstrings<std::runtime_error>(
      [&]() { (void)PnRNetlistReader::fromTOML(toml); },
      {"'from' edges", "same target node"});
}

TEST(PnRNetlistReaderFromTomlValidationErrorsTest, InvalidTomlSyntax) {
  const std::string toml = R"(
    [node
    n1 = "0,0"
  )";

  EXPECT_THROW(PnRNetlistReader::fromTOML(toml), std::runtime_error);
}

TEST(PnRNetlistReaderComplexNetlistTest, ParsesComplexNetlist) {
  const std::string toml = R"(
    [node]
    shim   = "0,0"
    mem    = "0,1"
    comp_1 = "0,2"
    comp_2 = "0,3"
    comp_3 = "0,4"
    comp_4 = "0,5"

    [edge]
    4 = [ "shim->mem,comp_1", [2, 7, 2], 7680 ]
    1 = [ "comp_1->comp_2",   [4],       1920 ]
    0 = [ "comp_2->comp_3",   [2],       1920 ]
    2 = [ "comp_3->comp_4",   [7, 2],    7680 ]
    3 = [ "mem->comp_4",      [7, 2],    7680 ]
    5 = [ "comp_4->comp_4",   [1],       7680 ]
    6 = [ "comp_4->mem",      [2],       7680 ]
    7 = [ "mem->shim",        [2],       7680 ]

    [linking]
    link_0 = "4->3"
    link_1 = "6->7"
  )";

  const auto reader = PnRNetlistReader::fromTOML(toml);
  const auto nodes = reader.getNodes();
  const auto edges = reader.getEdges();
  const auto linkings = reader.getLinkings();

  EXPECT_EQ(nodes.size(), 6U);
  EXPECT_EQ(edges.size(), 8U);
  EXPECT_EQ(linkings.size(), 2U);

  for (const auto &node : nodes) {
    EXPECT_TRUE(node.position.has_value());
  }

  for (const auto &edge : edges) {
    EXPECT_FALSE(edge.start_node.empty());
    EXPECT_FALSE(edge.target_nodes.empty());
  }

  const auto edge4 =
      std::ranges::find_if(edges, [](const auto &edge) { return edge.id == "4"; });
  ASSERT_NE(edge4, edges.end());
  EXPECT_EQ(edge4->start_node, "shim");
  EXPECT_EQ(edge4->target_nodes.size(), 2U);
  EXPECT_EQ(edge4->depth->size(), 3U);
}

TEST(PnRNetlistReaderAbsentSectionsTest,
     MissingNodeSectionAutoCreatesNodesFromEdges) {
  const std::string toml = R"(
    [edge]
    e1 = [ "n1->n2" ]

    [linking]
  )";

  const auto reader = PnRNetlistReader::fromTOML(toml);
  const auto nodes = reader.getNodes();
  const auto edges = reader.getEdges();
  const auto linkings = reader.getLinkings();

  EXPECT_EQ(nodes.size(), 2U);
  EXPECT_EQ(edges.size(), 1U);
  EXPECT_TRUE(linkings.empty());

  for (const auto &node : nodes) {
    EXPECT_FALSE(node.position.has_value());
  }
}

TEST(PnRNetlistReaderAbsentSectionsTest,
     MissingEdgeSectionFailsValidationForIsolatedNodes) {
  const std::string toml = R"(
    [node]
    n1 = "0,0"
    n2 = "0,1"

    [linking]
  )";

  gtest_helpers::ExpectThrowsWithSubstrings<std::runtime_error>(
      [&]() { (void)PnRNetlistReader::fromTOML(toml); }, {"not connected"});
}

TEST(PnRNetlistReaderAbsentSectionsTest, MissingLinkingSectionIsValid) {
  const std::string toml = R"(
    [node]
    n1 = "0,0"
    n2 = "0,1"

    [edge]
    e1 = [ "n1->n2" ]
  )";

  const auto reader = PnRNetlistReader::fromTOML(toml);
  const auto nodes = reader.getNodes();
  const auto edges = reader.getEdges();
  const auto linkings = reader.getLinkings();

  EXPECT_EQ(nodes.size(), 2U);
  EXPECT_EQ(edges.size(), 1U);
  EXPECT_TRUE(linkings.empty());
}

TEST(PnRNetlistReaderAbsentSectionsTest, AllSectionsAbsentYieldsEmptyNetlist) {
  const std::string toml = R"()";

  const auto reader = PnRNetlistReader::fromTOML(toml);
  EXPECT_TRUE(reader.getNodes().empty());
  EXPECT_TRUE(reader.getEdges().empty());
  EXPECT_TRUE(reader.getLinkings().empty());
}

TEST(PnRNetlistReaderAbsentSectionsTest, OnlyEdgeSectionPresent) {
  const std::string toml = R"(
    [edge]
    e1 = [ "n1->n2,n3", [2, 3, 4], 1920 ]
  )";

  const auto reader = PnRNetlistReader::fromTOML(toml);
  const auto nodes = reader.getNodes();
  const auto edges = reader.getEdges();
  const auto linkings = reader.getLinkings();

  EXPECT_EQ(nodes.size(), 3U);
  ASSERT_EQ(edges.size(), 1U);
  EXPECT_TRUE(linkings.empty());
  EXPECT_EQ(edges[0].id, "e1");
  EXPECT_EQ(edges[0].start_node, "n1");
  EXPECT_EQ(edges[0].target_nodes.size(), 2U);
  EXPECT_TRUE(edges[0].depth.has_value());
  EXPECT_TRUE(edges[0].width.has_value());
}

TEST(PnRNetlistReaderAbsentSectionsTest, EmptySectionsProduceValidNetlist) {
  const std::string toml = R"(
    [node]

    [edge]
    e1 = [ "n1->n2" ]

    [linking]
  )";

  const auto reader = PnRNetlistReader::fromTOML(toml);
  EXPECT_EQ(reader.getNodes().size(), 2U);
  EXPECT_EQ(reader.getEdges().size(), 1U);
  EXPECT_TRUE(reader.getLinkings().empty());
}

TEST(PnRNetlistReaderAbsentSectionsTest, NodeAndLinkingSectionsAbsent) {
  const std::string toml = R"(
    [edge]
    e1 = [ "a->b" ]
    e2 = [ "b->c" ]
  )";

  const auto reader = PnRNetlistReader::fromTOML(toml);
  EXPECT_EQ(reader.getNodes().size(), 3U);
  EXPECT_EQ(reader.getEdges().size(), 2U);
  EXPECT_TRUE(reader.getLinkings().empty());
}

TEST(PnRNetlistWriterBasicOperationsTest, AddSingleEdgeCreatesValidNetlist) {
  PnRNetlistWriter writer;
  writer.addNode({"n1", GridPosition{0, 0}});
  writer.addNode({"n2", GridPosition{0, 1}});
  writer.addEdge({"e1", "n1", {"n2"}, std::nullopt, std::nullopt});

  const std::string toml = writer.toTOML();
  EXPECT_NE(toml.find("n1 = \"0,0\""), std::string::npos);
  EXPECT_NE(toml.find("n2 = \"0,1\""), std::string::npos);
  EXPECT_NE(toml.find("e1 = [ \"n1->n2\" ]"), std::string::npos);
}

TEST(PnRNetlistWriterBasicOperationsTest,
     AddEdgeAutoCreatesNodesWithoutPositions) {
  PnRNetlistWriter writer;
  writer.addEdge({"e1", "n1", {"n2"}, std::nullopt, std::nullopt});

  const std::string toml = writer.toTOML();
  EXPECT_EQ(toml.find("n1 = "), std::string::npos);
  EXPECT_EQ(toml.find("n2 = "), std::string::npos);
  EXPECT_NE(toml.find("e1 = [ \"n1->n2\" ]"), std::string::npos);
}

TEST(PnRNetlistWriterBasicOperationsTest, AddEdgeWithDepthAndWidth) {
  PnRNetlistWriter writer;
  writer.addEdge(
      {"e1", "n1", {"n2", "n3"}, std::vector<size_t>{2, 3, 4}, 1920});

  const std::string toml = writer.toTOML();
  EXPECT_NE(toml.find("e1 = [ \"n1->n2,n3\", [ 2, 3, 4 ], 1920 ]"),
            std::string::npos);
}

TEST(PnRNetlistWriterBasicOperationsTest, AddLinking) {
  PnRNetlistWriter writer;
  writer.addEdge({"e1", "n1", {"n2"}, std::vector<size_t>{2, 2}, 1920});
  writer.addEdge({"e2", "n2", {"n3"}, std::vector<size_t>{2, 2}, 1920});
  writer.addLinking({"link1", {"e1"}, {"e2"}});

  const std::string toml = writer.toTOML();
  EXPECT_NE(toml.find("link1 = \"e1->e2\""), std::string::npos);
}

TEST(PnRNetlistWriterBasicOperationsTest, DuplicateNodeThrowsException) {
  PnRNetlistWriter writer;
  writer.addNode({"n1", GridPosition{0, 0}});

  gtest_helpers::ExpectThrowsWithSubstrings<std::runtime_error>(
      [&]() { writer.addNode({"n1", GridPosition{1, 1}}); },
      {"already exists"});
}

TEST(PnRNetlistWriterBasicOperationsTest, AddRouteInformation) {
  PnRNetlistWriter writer;
  writer.addEdge(
      {"e1", "n1", {"n2", "n3"}, std::vector<size_t>{2, 2, 2}, 1920});

  PnRNetlistFormat::RouteInfo route;
  route.edge_id = "e1";
  route.type = "circuit_switch";
  route.paths.insert_or_assign(
      "n2", std::vector{GridPosition{1, 1}, GridPosition{1, 2}});
  route.paths.insert_or_assign(
      "n3", std::vector{GridPosition{1, 1}, GridPosition{2, 1}});

  writer.addRoute(route);

  const std::string toml = writer.toTOML();
  EXPECT_NE(toml.find("[route]"), std::string::npos);
  EXPECT_TRUE(
      toml.find("e1 = [ \"circuit_switch\", { n2 = [ \"1,1\", \"1,2\" ], "
                "n3 = [ \"1,1\", \"2,1\" ] } ]") != std::string::npos ||
      toml.find("e1 = [ \"circuit_switch\", { n3 = [ \"1,1\", \"2,1\" ], "
                "n2 = [ \"1,1\", \"1,2\" ] } ]") != std::string::npos);
}

TEST(PnRNetlistWriterBasicOperationsTest, MultipleRoutes) {
  PnRNetlistWriter writer;
  writer.addEdge({"e1", "n1", {"n2"}, std::vector<size_t>{2, 2}, 1920});
  writer.addEdge({"e2", "n2", {"n3"}, std::vector<size_t>{2, 2}, 1920});
  writer.addEdge({"e3", "n3", {"n3"}, std::vector<size_t>{2, 2}, 1920});

  PnRNetlistFormat::RouteInfo route1;
  route1.edge_id = "e1";
  route1.type = "packet_switch";
  route1.paths.insert_or_assign("n2", std::vector{GridPosition{0, 1}});

  PnRNetlistFormat::RouteInfo route2;
  route2.edge_id = "e2";
  route2.type = "neighbor_sharing";
  route2.paths.insert_or_assign("n3", std::vector{GridPosition{1, 2}});

  PnRNetlistFormat::RouteInfo route3;
  route3.edge_id = "e3";
  route3.type = "intra_tile";
  route3.paths.insert_or_assign("n3", std::vector{GridPosition{1, 2}});

  writer.addRoute(route1);
  writer.addRoute(route2);
  writer.addRoute(route3);

  const std::string toml = writer.toTOML();
  EXPECT_NE(toml.find("e1 = [ \"packet_switch\", { n2 = [ \"0,1\" ] } ]"),
            std::string::npos);
  EXPECT_NE(
      toml.find("e2 = [ \"neighbor_sharing\", { n3 = [ \"1,2\" ] } ]"),
      std::string::npos);
  EXPECT_NE(toml.find("e3 = [ \"intra_tile\", { n3 = [ \"1,2\" ] } ]"),
            std::string::npos);
}

TEST(PnRNetlistWriterValidationTest, IsolatedNodeFailsValidation) {
  PnRNetlistWriter writer;
  writer.addNode({"isolated", GridPosition{0, 0}});
  writer.addNode({"n1", GridPosition{0, 1}});
  writer.addNode({"n2", GridPosition{0, 2}});
  writer.addEdge({"e1", "n1", {"n2"}, std::nullopt, std::nullopt});

  gtest_helpers::ExpectThrowsWithSubstrings<std::runtime_error>(
      [&]() { (void)writer.toTOML(); }, {"not connected"});
}

TEST(PnRNetlistWriterValidationTest, DepthWithoutWidthFailsValidation) {
  PnRNetlistWriter writer;
  writer.addEdge(
      {"e1", "n1", {"n2"}, std::vector<size_t>{2, 2}, std::nullopt});

  gtest_helpers::ExpectThrowsWithSubstrings<std::runtime_error>(
      [&]() { (void)writer.toTOML(); }, {"depth specified but missing width"});
}

TEST(PnRNetlistWriterValidationTest, WidthWithoutDepthFailsValidation) {
  PnRNetlistWriter writer;
  writer.addEdge({"e1", "n1", {"n2"}, std::nullopt, 1920});

  gtest_helpers::ExpectThrowsWithSubstrings<std::runtime_error>(
      [&]() { (void)writer.toTOML(); }, {"width specified but missing depth"});
}

TEST(PnRNetlistRoundTripConversionTest, RegeneratedTomlCanBeParsedAgain) {
  const std::string original_toml = R"(
    [node]
    n1 = "0,0"
    n2 = "0,1"
    n3 = "0,2"

    [edge]
    e1 = [ "n1->n2", [2], 1920 ]
    e2 = [ "n2->n3", [2], 1920 ]

    [linking]
    link1 = "e1->e2"
  )";

  const auto reader = PnRNetlistReader::fromTOML(original_toml);

  PnRNetlistWriter writer;
  for (const auto &node : reader.getNodes()) {
    writer.addNode(node);
  }
  for (const auto &edge : reader.getEdges()) {
    writer.addEdge(edge);
  }
  for (const auto &linking : reader.getLinkings()) {
    writer.addLinking(linking);
  }

  const std::string regenerated_toml = writer.toTOML();

  const auto reader2 = PnRNetlistReader::fromTOML(regenerated_toml);
  EXPECT_EQ(reader2.getNodes().size(), reader.getNodes().size());
  EXPECT_EQ(reader2.getEdges().size(), reader.getEdges().size());
  EXPECT_EQ(reader2.getLinkings().size(), reader.getLinkings().size());
}
