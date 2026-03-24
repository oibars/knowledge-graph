"""
Semantic Linker Service
Automatically creates knowledge graph links using embeddings and LLM-based analysis.
"""

import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
import structlog

from knowledge_graph.models import Entity, Relation, RELATION_TYPES
from knowledge_graph.services.graph_store import KnowledgeGraphStore

logger = structlog.get_logger()


class SemanticLinker:
    """
    Creates semantic links between entities in the knowledge graph.
    
    Uses embeddings for similarity-based linking and can leverage LLM
    for concept extraction and relationship inference.
    """
    
    def __init__(
        self,
        graph_store: KnowledgeGraphStore,
        embedding_model=None,
        llm_client=None,
        similarity_threshold: float = 0.7
    ):
        self.graph = graph_store
        self.embedding_model = embedding_model
        self.llm_client = llm_client
        self.similarity_threshold = similarity_threshold
    
    def _compute_embedding(self, text: str) -> Optional[List[float]]:
        """Compute embedding for text using the configured model."""
        if not self.embedding_model:
            return None
        
        try:
            # Assuming embedding_model is compatible with sentence-transformers interface
            embedding = self.embedding_model.encode(text)
            return embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
        except Exception as e:
            logger.error("Failed to compute embedding", text=text[:50], error=str(e))
            return None
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(v1, v2) / (norm1 * norm2))
    
    def link_file_to_concepts(
        self,
        file_entity_id: str,
        extract_concepts: bool = True
    ) -> List[Relation]:
        """
        Link a file entity to concept entities.
        
        If extract_concepts is True, will use LLM to extract concepts
        from the file content and create new concept entities.
        """
        file_entity = self.graph.get_entity(file_entity_id)
        if not file_entity:
            logger.warning("File entity not found", file_id=file_entity_id)
            return []
        
        relations = []
        
        # Find existing concept entities with similar embeddings
        if file_entity.embedding:
            similar = self.graph.find_similar_entities(file_entity_id, limit=10)
            for similar_entity, similarity in similar:
                if similar_entity.label == "Concept" and similarity >= self.similarity_threshold:
                    relation_id = f"rel_{file_entity_id}_{similar_entity.id}_implements"
                    relation = Relation(
                        id=relation_id,
                        source_id=file_entity_id,
                        target_id=similar_entity.id,
                        relation_type="implements",
                        strength=similarity,
                        connection_reason=f"Semantic similarity: {similarity:.2f}",
                        is_auto_generated=True
                    )
                    self.graph.add_relation(relation)
                    relations.append(relation)
        
        # Extract and link new concepts if LLM available
        if extract_concepts and self.llm_client and file_entity.description:
            concepts = self._extract_concepts_from_text(file_entity.description)
            for concept_name in concepts:
                # Check if concept already exists
                existing = self.graph.search_entities(concept_name, label="Concept", limit=1)
                
                if existing:
                    concept_entity = existing[0]
                else:
                    # Create new concept entity
                    concept_id = f"concept_{concept_name.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                    concept_embedding = self._compute_embedding(concept_name)
                    
                    concept_entity = Entity(
                        id=concept_id,
                        label="Concept",
                        name=concept_name,
                        description=f"Concept extracted from {file_entity.name}",
                        embedding=concept_embedding,
                        source_file_path=file_entity.source_file_path,
                        is_auto_generated=True
                    )
                    self.graph.add_entity(concept_entity)
                
                # Create relation
                relation_id = f"rel_{file_entity_id}_{concept_entity.id}_implements"
                relation = Relation(
                    id=relation_id,
                    source_id=file_entity_id,
                    target_id=concept_entity.id,
                    relation_type="implements",
                    strength=0.8,
                    connection_reason="LLM-extracted concept",
                    is_auto_generated=True
                )
                self.graph.add_relation(relation)
                relations.append(relation)
        
        logger.info(
            "Linked file to concepts",
            file_id=file_entity_id,
            relations_created=len(relations)
        )
        return relations
    
    def _extract_concepts_from_text(self, text: str) -> List[str]:
        """Extract key concepts from text using LLM."""
        if not self.llm_client:
            return []
        
        try:
            # This is a simplified version - in production would use proper async
            prompt = f"""Extract the 3-5 most important concepts from the following text.
Return only a comma-separated list of concept names.

Text: {text[:1000]}

Concepts:"""
            
            # Placeholder for actual LLM call
            # response = await self.llm_client.chat_completion(...)
            # For now, return empty list
            return []
            
        except Exception as e:
            logger.error("Failed to extract concepts", error=str(e))
            return []
    
    def link_related_tasks(self, task_entity_id: str) -> List[Relation]:
        """
        Find and create links between related tasks.
        
        Tasks are related if they:
        - Share similar concepts/tags
        - Have similar embeddings
        - Are in the same folder/project
        """
        task_entity = self.graph.get_entity(task_entity_id)
        if not task_entity or task_entity.label != "Task":
            return []
        
        relations = []
        
        # Find other tasks
        other_tasks = self.graph.find_by_label("Task")
        
        for other_task in other_tasks:
            if other_task.id == task_entity_id:
                continue
            
            # Calculate similarity score
            similarity = 0.0
            
            # Tag overlap
            if task_entity.tags and other_task.tags:
                common_tags = set(task_entity.tags) & set(other_task.tags)
                if common_tags:
                    similarity += len(common_tags) * 0.2
            
            # Embedding similarity
            if task_entity.embedding and other_task.embedding:
                embedding_sim = self._cosine_similarity(
                    task_entity.embedding,
                    other_task.embedding
                )
                similarity += embedding_sim * 0.5
            
            # Same source app/folder
            if task_entity.source_app and task_entity.source_app == other_task.source_app:
                similarity += 0.1
            
            if task_entity.source_file_path and other_task.source_file_path:
                path1 = task_entity.source_file_path
                path2 = other_task.source_file_path
                if path1.parent == path2.parent:
                    similarity += 0.2
            
            # Create relation if similarity is high enough
            if similarity >= self.similarity_threshold:
                relation_id = f"rel_{task_entity_id}_{other_task.id}_similar_to"
                relation = Relation(
                    id=relation_id,
                    source_id=task_entity_id,
                    target_id=other_task.id,
                    relation_type="similar_to",
                    strength=min(similarity, 1.0),
                    connection_reason=f"Task similarity: {similarity:.2f}",
                    is_auto_generated=True,
                    bidirectional=True
                )
                self.graph.add_relation(relation)
                relations.append(relation)
        
        return relations
    
    def discover_implicit_links(
        self,
        entity_id: str,
        max_depth: int = 2
    ) -> List[Tuple[Relation, float]]:
        """
        Discover implicit connections through multi-hop traversal.
        
        Returns list of (relation, confidence) tuples for suggested new links.
        """
        entity = self.graph.get_entity(entity_id)
        if not entity:
            return []
        
        discovered = []
        visited = {entity_id}
        current_level = {entity_id}
        
        for depth in range(1, max_depth + 1):
            next_level = set()
            
            for node_id in current_level:
                # Get neighbors
                neighbors = self.graph.get_neighbors(node_id, depth=1)
                
                for depth_key, neighbor_list in neighbors.items():
                    if depth_key != 1:
                        continue
                    
                    for neighbor in neighbor_list:
                        if neighbor.id not in visited:
                            visited.add(neighbor.id)
                            next_level.add(neighbor.id)
                            
                            # Calculate discovery confidence
                            confidence = 1.0 / depth  # Decreases with distance
                            
                            # Check if link already exists
                            existing = self.graph.get_entity_relations(entity_id)
                            existing_targets = {r.target_id for r in existing if r.source_id == entity_id}
                            existing_targets |= {r.source_id for r in existing if r.target_id == entity_id}
                            
                            if neighbor.id not in existing_targets:
                                # Suggest new implicit link
                                relation_id = f"rel_implicit_{entity_id}_{neighbor.id}_{depth}hop"
                                relation = Relation(
                                    id=relation_id,
                                    source_id=entity_id,
                                    target_id=neighbor.id,
                                    relation_type="related_to",
                                    strength=confidence * 0.5,
                                    connection_reason=f"Discovered via {depth}-hop traversal",
                                    is_auto_generated=True
                                )
                                discovered.append((relation, confidence))
            
            current_level = next_level
            if not current_level:
                break
        
        # Sort by confidence
        discovered.sort(key=lambda x: x[1], reverse=True)
        return discovered
    
    def suggest_connections(
        self,
        entity_id: str,
        limit: int = 5
    ) -> List[Tuple[Entity, float, str]]:
        """
        Suggest potential connections for an entity.
        
        Returns list of (entity, score, reason) tuples.
        """
        entity = self.graph.get_entity(entity_id)
        if not entity:
            return []
        
        suggestions = []
        
        # 1. Embedding-based similarity
        if entity.embedding:
            similar_entities = self.graph.find_similar_entities(entity_id, limit=limit * 2)
            for similar_entity, similarity in similar_entities:
                if similar_entity.id != entity_id:
                    suggestions.append((similar_entity, similarity, "Semantic similarity"))
        
        # 2. Content-based matching (name/description overlap)
        entity_words = set()
        if entity.name:
            entity_words |= set(entity.name.lower().split())
        if entity.description:
            entity_words |= set(entity.description.lower().split())
        
        for other in self.graph.get_all_entities():
            if other.id == entity_id:
                continue
            
            other_words = set()
            if other.name:
                other_words |= set(other.name.lower().split())
            if other.description:
                other_words |= set(other.description.lower().split())
            
            common = entity_words & other_words
            if len(common) >= 3:  # At least 3 common words
                score = min(len(common) / 10, 0.9)  # Cap at 0.9
                suggestions.append((other, score, f"{len(common)} common terms"))
        
        # 3. Tag overlap
        if entity.tags:
            for other in self.graph.find_by_label(entity.label):
                if other.id == entity_id:
                    continue
                
                if other.tags:
                    common_tags = set(entity.tags) & set(other.tags)
                    if common_tags:
                        score = len(common_tags) * 0.15
                        suggestions.append((other, min(score, 0.85), f"Shared tags: {', '.join(common_tags)}"))
        
        # Remove duplicates, keeping highest score
        seen = {}
        for entity, score, reason in suggestions:
            if entity.id not in seen or seen[entity.id][1] < score:
                seen[entity.id] = (entity, score, reason)
        
        # Sort by score and return top N
        result = sorted(seen.values(), key=lambda x: x[1], reverse=True)[:limit]
        return result
    
    def auto_link_entity(
        self,
        entity_id: str,
        create_links: bool = False
    ) -> Dict[str, Any]:
        """
        Automatically discover and optionally create links for an entity.
        
        Returns summary of discovered links.
        """
        discovered = {
            "concepts": [],
            "related_tasks": [],
            "implicit_links": [],
            "suggestions": []
        }
        
        entity = self.graph.get_entity(entity_id)
        if not entity:
            return discovered
        
        # Link to concepts
        if entity.label == "File":
            concept_relations = self.link_file_to_concepts(entity_id)
            discovered["concepts"] = [r.target_id for r in concept_relations]
        
        # Link related tasks
        if entity.label == "Task":
            task_relations = self.link_related_tasks(entity_id)
            discovered["related_tasks"] = [r.target_id for r in task_relations]
        
        # Discover implicit links
        implicit = self.discover_implicit_links(entity_id)
        discovered["implicit_links"] = [
            {"target": r.target_id, "confidence": c}
            for r, c in implicit
        ]
        
        if create_links:
            for relation, confidence in implicit:
                if confidence >= 0.5:
                    self.graph.add_relation(relation)
        
        # Get suggestions
        suggestions = self.suggest_connections(entity_id, limit=5)
        discovered["suggestions"] = [
            {"entity_id": e.id, "entity_name": e.name, "score": s, "reason": r}
            for e, s, r in suggestions
        ]
        
        logger.info(
            "Auto-linked entity",
            entity_id=entity_id,
            concepts=len(discovered["concepts"]),
            related_tasks=len(discovered["related_tasks"]),
            implicit=len(discovered["implicit_links"])
        )
        
        return discovered
    
    def batch_link_entities(
        self,
        entity_ids: List[str],
        create_links: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        Automatically link multiple entities.
        
        Returns mapping of entity_id to discovery results.
        """
        results = {}
        
        for entity_id in entity_ids:
            try:
                result = self.auto_link_entity(entity_id, create_links=create_links)
                results[entity_id] = result
            except Exception as e:
                logger.error("Failed to auto-link entity", entity_id=entity_id, error=str(e))
                results[entity_id] = {"error": str(e)}
        
        return results
    
    def create_embedding_for_entity(self, entity_id: str) -> bool:
        """Generate and store embedding for an entity."""
        entity = self.graph.get_entity(entity_id)
        if not entity:
            return False
        
        # Build text representation
        text_parts = [entity.name]
        if entity.description:
            text_parts.append(entity.description)
        if entity.tags:
            text_parts.extend(entity.tags)
        
        text = " ".join(text_parts)
        
        embedding = self._compute_embedding(text)
        if embedding:
            entity.embedding = embedding
            self.graph.update_entity(entity)
            return True
        
        return False
    
    def bulk_create_embeddings(self, label: Optional[str] = None) -> int:
        """Create embeddings for all entities (optionally filtered by label)."""
        if label:
            entities = self.graph.find_by_label(label)
        else:
            entities = self.graph.get_all_entities()
        
        count = 0
        for entity in entities:
            if not entity.embedding:
                if self.create_embedding_for_entity(entity.id):
                    count += 1
        
        logger.info("Bulk embeddings created", count=count, label=label)
        return count
