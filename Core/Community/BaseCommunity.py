from abc import ABC, abstractmethod
from Core.Common.Logger import logger


class BaseCommunity(ABC):
    """Base community class definition."""

    def __init__(self, llm, enforce_sub_communities, namespace):
        self.llm = llm
        self.enforce_sub_communities = enforce_sub_communities
        self.namespace = namespace

        
    async def generate_community_report(self, graph, force=False):
        """
            Generates a community report based on the provided graph.

            This function first attempts to load an existing community report. If the report does not exist or if the `force` flag is set to True, it will generate a new community report from the provided graph. After generating the report, it persists the report to a file.

            Args:
                graph: The graph data structure used to generate the community report.
                force (bool): If True, forces the generation of a new community report even if one already exists. Defaults to False.

        """
        # Try to load the community report
        logger.info("Generating community report...")

        is_exist = await self._load_community_report(graph,force)
        if force or not is_exist:
            # Generate the community report
            await self._generate_community_report(graph)
            # Persist the community report
            await self._persist_community()
        logger.info("✅ [Community Report]  Finished")

    async def cluster(self, **kwargs):
        """
          Clusters the input graph .

          This function first attempts to load an existing cluster map. If the cluster map does not exist or if the `force` flag is set to True, it will perform clustering on the data. After clustering, it persists the cluster map to a file.

          Args:
              **kwargs: Additional keyword arguments that may include parameters for clustering.
                  - force (bool): If True, forces the clustering process even if a cluster map already exists. Defaults to False.
          """
        logger.info("Starting build community of the given graph")
        logger.start("Clustering nodes")
        force = kwargs.pop('force', False)
        # Try to load the community <-> node map
        is_exist = await self._load_cluster_map(force)
        if force or not is_exist:
           
            # Clustering the graph and generate the community <-> node map
            await self.clustering(**kwargs)
            # Persist the community <-> node map
            await self._persist_cluster_map()

    @abstractmethod
    async def _generate_community_report(self, graph):
        pass

    @abstractmethod
    async def clustering(self, **kwargs):
        pass

    @abstractmethod
    async def _load_community_report(self, force):
        pass

    @abstractmethod
    async def _persist_community(self):
        pass

    @abstractmethod
    async def _load_cluster_map(self, force):
        pass

    @abstractmethod
    async def _persist_cluster_map(self):
        pass
