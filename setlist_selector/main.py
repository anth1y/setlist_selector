#!/usr/bin/env python3
"""
Setlist Selector - AI-powered setlist creator MCP server
Creates intelligent setlists by integrating Spotify, Last.fm, and AI curation
"""

import asyncio
import json
import logging
import base64
import sys
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
from mcp.types import Resource, Tool, TextContent, ImageContent, EmbeddedResource
from pydantic import AnyUrl
import httpx
import anthropic
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("setlist_selector")


class SetlistCreatorAgent:
    """Main agent class that uses Claude for intelligent setlist curation"""

    def __init__(self) -> None:
        self.anthropic_client = anthropic.AsyncAnthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )

    async def curate_setlist_with_ai(
        self, prompt: str, existing_songs: Optional[List[Dict[Any, Any]]] = None
    ) -> Dict[Any, Any]:
        """Use Claude to curate a setlist based on user prompt"""

        existing_context = ""
        if existing_songs:
            songs_text = "\n".join(
                [
                    f"- {song.get('name', 'Unknown')} by {', '.join([artist['name'] for artist in song.get('artists', [])])}"
                    for song in existing_songs[:10]
                ]
            )
            existing_context = f"\n\nExisting songs for reference:\n{songs_text}"

        system_prompt = """You are an expert music curator with deep knowledge of music genres, artists, and songs across all eras.
        You create thoughtful setlists that match the user's mood, activity, or theme perfectly.

        Always respond with a valid JSON object containing exactly these fields:
        1. "setlist_name": Creative name for the setlist (string)
        2. "description": Brief description of the setlist's vibe (string)
        3. "search_queries": Array of search terms to find songs - use "artist song" format for best results (array of strings)
        4. "mood_tags": Array of mood/genre tags (array of strings)
        5. "target_length": Suggested number of songs between 10-50 (integer)

        Make the search queries specific enough to find real songs but diverse enough for a good mix.
        Focus on well-known artists and popular songs that are likely to be on Spotify."""

        user_message = f"Create a setlist based on: {prompt}{existing_context}"

        try:
            response = await self.anthropic_client.messages.create(
                model="claude-3-5-haiku-20241022",  # Fast and cost-effective model
                max_tokens=1000,
                temperature=0.7,  # Higher creativity for music curation
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )

            # Get the text content from the response
            content = ""
            for block in response.content:
                if hasattr(block, 'text'):
                    content = block.text.strip()
                    break

            # Clean up JSON formatting if wrapped in markdown
            if content.startswith("```json"):
                content = content[7:-3]
            elif content.startswith("```"):
                content = content[3:-3]

            result: Dict[Any, Any] = json.loads(content)

            # Validate required fields
            required_fields = [
                "setlist_name",
                "description",
                "search_queries",
                "mood_tags",
                "target_length",
            ]
            for field in required_fields:
                if field not in result:
                    result[field] = self._get_default_value(field, prompt)

            # Ensure target_length is reasonable
            if (
                not isinstance(result["target_length"], int)
                or result["target_length"] < 5
                or result["target_length"] > 50
            ):
                result["target_length"] = 20

            return result

        except Exception as e:
            logger.error(f"Claude setlist curation failed: {e}")
            return self._get_fallback_setlist(prompt)

    def _get_default_value(self, field: str, prompt: str) -> Any:
        """Get default values for missing fields"""
        defaults: Dict[str, Any] = {
            "setlist_name": f"Custom Setlist - {prompt[:20]}...",
            "description": f"A curated setlist based on: {prompt}",
            "search_queries": ["popular songs", "hit music", "trending tracks"],
            "mood_tags": ["general", "popular"],
            "target_length": 20,
        }
        return defaults.get(field, "")

    def _get_fallback_setlist(self, prompt: str) -> Dict[Any, Any]:
        """Fallback setlist when Claude fails"""
        return {
            "setlist_name": f"AI Setlist - {prompt[:20]}",
            "description": f"A curated setlist for: {prompt}",
            "search_queries": ["popular songs", "trending music", "hit songs"],
            "mood_tags": ["general"],
            "target_length": 15,
        }


class SpotifyService:
    """Spotify Web API integration"""

    def __init__(self) -> None:
        self.client_id = os.getenv("SPOTIFY_CLIENT_ID")
        self.client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
        self.access_token: Optional[str] = None
        self.token_expires_at: float = 0

    async def _get_access_token(self) -> str:
        """Get Spotify access token using client credentials flow"""
        if self.access_token and datetime.now().timestamp() < self.token_expires_at:
            return self.access_token

        auth_str = f"{self.client_id}:{self.client_secret}"
        auth_bytes = auth_str.encode("ascii")
        auth_b64 = base64.b64encode(auth_bytes).decode("ascii")

        headers = {
            "Authorization": f"Basic {auth_b64}",
            "Content-Type": "application/x-www-form-urlencoded",
        }

        data = "grant_type=client_credentials"

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://accounts.spotify.com/api/token", headers=headers, content=data
            )
            response.raise_for_status()
            token_data = response.json()

            self.access_token = token_data["access_token"]
            self.token_expires_at = (
                datetime.now().timestamp() + token_data.get("expires_in", 3600) - 60
            )

            return self.access_token

    async def search_tracks(self, query: str, limit: int = 10) -> List[Dict[Any, Any]]:
        """Search for tracks on Spotify"""
        token = await self._get_access_token()
        headers = {"Authorization": f"Bearer {token}"}

        params: Dict[str, Union[str, int]] = {"q": query, "type": "track", "limit": min(limit, 50)}

        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.spotify.com/v1/search", headers=headers, params=params
            )
            response.raise_for_status()
            data = response.json()
            tracks: List[Dict[Any, Any]] = data.get("tracks", {}).get("items", [])
            return tracks

    async def get_recommendations(
        self,
        seed_tracks: Optional[List[str]] = None,
        seed_artists: Optional[List[str]] = None,
        target_features: Optional[Dict[Any, Any]] = None,
        limit: int = 20,
    ) -> List[Dict[Any, Any]]:
        """Get track recommendations from Spotify"""
        token = await self._get_access_token()
        headers = {"Authorization": f"Bearer {token}"}

        params: Dict[str, Union[str, int, float]] = {"limit": min(limit, 100)}

        if seed_tracks:
            params["seed_tracks"] = ",".join(seed_tracks[:5])  # Max 5 seeds
        if seed_artists:
            params["seed_artists"] = ",".join(seed_artists[:5])

        # Add target audio features if provided
        if target_features:
            for feature, value in target_features.items():
                if feature in ["danceability", "energy", "valence", "tempo"]:
                    params[f"target_{feature}"] = value

        # If no seeds provided, use popular genres
        if not seed_tracks and not seed_artists:
            params["seed_genres"] = "pop,rock,hip-hop"

        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.spotify.com/v1/recommendations",
                headers=headers,
                params=params,
            )
            response.raise_for_status()
            data = response.json()
            tracks: List[Dict[Any, Any]] = data.get("tracks", [])
            return tracks


class LastFmService:
    """Last.fm API integration for music data and recommendations"""

    def __init__(self) -> None:
        self.api_key = os.getenv("LASTFM_API_KEY")
        self.base_url = "http://ws.audioscrobbler.com/2.0/"

    async def get_similar_artists(self, artist: str, limit: int = 10) -> List[Dict[Any, Any]]:
        """Get similar artists from Last.fm"""
        params: Dict[str, Union[str, int]] = {
            "method": "artist.getSimilar",
            "artist": artist,
            "api_key": self.api_key or "",
            "format": "json",
            "limit": limit,
        }

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(self.base_url, params=params)
                response.raise_for_status()
                data = response.json()

                similar_artists = data.get("similarartists", {}).get("artist", [])
                if isinstance(similar_artists, dict):
                    similar_artists = [similar_artists]

                # Ensure we return the correct type
                result: List[Dict[Any, Any]] = similar_artists
                return result

            except Exception as e:
                logger.warning(f"Last.fm API error for similar artists: {e}")
                return []

    async def get_top_tracks_by_tag(self, tag: str, limit: int = 50) -> List[Dict[Any, Any]]:
        """Get top tracks for a specific tag/genre"""
        params: Dict[str, Union[str, int]] = {
            "method": "tag.getTopTracks",
            "tag": tag,
            "api_key": self.api_key or "",
            "format": "json",
            "limit": limit,
        }

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(self.base_url, params=params)
                response.raise_for_status()
                data = response.json()

                tracks = data.get("tracks", {}).get("track", [])
                if isinstance(tracks, dict):
                    tracks = [tracks]

                # Ensure we return the correct type
                result: List[Dict[Any, Any]] = tracks
                return result

            except Exception as e:
                logger.warning(f"Last.fm API error for top tracks: {e}")
                return []


# Initialize services
agent = SetlistCreatorAgent()
spotify_service = SpotifyService()
lastfm_service = LastFmService()

# Create MCP Server
server = Server("setlist_selector")


@server.list_tools()  # type: ignore[misc]
async def handle_list_tools() -> List[Tool]:
    """List available tools for the MCP client"""
    return [
        Tool(
            name="create_ai_setlist",
            description="Create an intelligent setlist based on natural language description using AI",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Describe the setlist you want (e.g., 'upbeat workout music', 'chill coffee shop vibes', 'high-energy rock concert opener')",
                    },
                    "include_recommendations": {
                        "type": "boolean",
                        "description": "Whether to include Spotify recommendations (default: true)",
                    },
                },
                "required": ["prompt"],
            },
        ),
        Tool(
            name="search_songs",
            description="Search for songs on Spotify",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (artist name, song title, or keywords)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of results to return (1-50, default: 10)",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="get_similar_music",
            description="Find similar artists and songs based on a given artist",
            inputSchema={
                "type": "object",
                "properties": {
                    "artist": {
                        "type": "string",
                        "description": "Artist name to find similar music for",
                    },
                    "include_tracks": {
                        "type": "boolean",
                        "description": "Whether to include track recommendations (default: true)",
                    },
                },
                "required": ["artist"],
            },
        ),
        Tool(
            name="discover_by_mood",
            description="Discover music by mood or genre using Last.fm tags",
            inputSchema={
                "type": "object",
                "properties": {
                    "mood_or_genre": {
                        "type": "string",
                        "description": "Mood or genre tag (e.g., 'happy', 'chill', 'electronic', 'indie')",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of tracks to discover (default: 20)",
                    },
                },
                "required": ["mood_or_genre"],
            },
        ),
    ]


@server.call_tool()  # type: ignore[misc]
async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool execution"""

    if name == "create_ai_setlist":
        try:
            prompt = arguments["prompt"]
            include_recommendations = arguments.get("include_recommendations", True)

            # Use Claude to analyze prompt and create setlist structure
            setlist_plan = await agent.curate_setlist_with_ai(prompt)

            # Search for songs based on AI suggestions
            all_tracks = []
            search_queries = setlist_plan.get("search_queries", [])

            for query in search_queries[:10]:  # Limit searches to prevent rate limits
                try:
                    tracks = await spotify_service.search_tracks(query, limit=5)
                    all_tracks.extend(tracks)
                except Exception as e:
                    logger.warning(f"Search failed for '{query}': {e}")
                    continue

            # Get Spotify recommendations if requested
            recommendations = []
            if include_recommendations and all_tracks:
                try:
                    # Use first few tracks as seeds for recommendations
                    seed_track_ids = [track["id"] for track in all_tracks[:3]]
                    recommendations = await spotify_service.get_recommendations(
                        seed_tracks=seed_track_ids, limit=10
                    )
                except Exception as e:
                    logger.warning(f"Recommendations failed: {e}")

            # Combine and deduplicate tracks
            combined_tracks = all_tracks + recommendations
            unique_tracks = []
            seen_ids = set()

            for track in combined_tracks:
                if track["id"] not in seen_ids:
                    unique_tracks.append(track)
                    seen_ids.add(track["id"])

            # Limit to target length
            target_length = setlist_plan.get("target_length", 20)
            final_tracks = unique_tracks[:target_length]

            # Format response
            result = f"""# 🎵 {setlist_plan['setlist_name']}

**Description:** {setlist_plan['description']}

**Mood Tags:** {', '.join(setlist_plan.get('mood_tags', []))}

## Setlist ({len(final_tracks)} songs)

"""

            for i, track in enumerate(final_tracks, 1):
                artists = ", ".join(
                    [artist["name"] for artist in track.get("artists", [])]
                )
                duration_ms = track.get("duration_ms", 0)
                duration_min = duration_ms // 60000
                duration_sec = (duration_ms % 60000) // 1000

                result += f"{i:2d}. **{track['name']}** by {artists} ({duration_min}:{duration_sec:02d})\n"
                result += f"    🎧 [Listen on Spotify]({track['external_urls']['spotify']})\n\n"

            if not final_tracks:
                result += "⚠️ No tracks found. Try a different description or check your API keys.\n"

            return [TextContent(type="text", text=result)]

        except Exception as e:
            logger.error(f"Error creating setlist: {e}")
            return [
                TextContent(type="text", text=f"❌ Error creating setlist: {str(e)}")
            ]

    elif name == "search_songs":
        try:
            query = arguments["query"]
            limit = arguments.get("limit", 10)

            tracks = await spotify_service.search_tracks(query, limit)

            if not tracks:
                return [TextContent(type="text", text=f"No tracks found for '{query}'")]

            result = f"# 🔍 Search Results for '{query}'\n\n"

            for i, track in enumerate(tracks, 1):
                artists = ", ".join(
                    [artist["name"] for artist in track.get("artists", [])]
                )
                album = track.get("album", {}).get("name", "Unknown Album")

                result += f"{i}. **{track['name']}** by {artists}\n"
                result += f"   Album: {album}\n"
                result += f"   🎧 [Spotify]({track['external_urls']['spotify']})\n\n"

            return [TextContent(type="text", text=result)]

        except Exception as e:
            logger.error(f"Error searching songs: {e}")
            return [TextContent(type="text", text=f"❌ Error searching: {str(e)}")]

    elif name == "get_similar_music":
        try:
            artist = arguments["artist"]
            include_tracks = arguments.get("include_tracks", True)

            # Get similar artists from Last.fm
            similar_artists = await lastfm_service.get_similar_artists(artist, limit=10)

            result = f"# 🎯 Artists Similar to {artist}\n\n"

            if similar_artists:
                for i, similar_artist in enumerate(similar_artists[:5], 1):
                    name = similar_artist.get("name", "Unknown")
                    match_score = similar_artist.get("match", "0")
                    match_percent = (
                        float(match_score) * 100 if match_score != "0" else 0
                    )

                    result += f"{i}. **{name}** ({match_percent:.0f}% match)\n"

                # If requested, get some tracks from similar artists
                if include_tracks and similar_artists:
                    result += "\n## 🎵 Recommended Tracks\n\n"

                    for similar_artist in similar_artists[:3]:
                        artist_name = similar_artist.get("name", "")
                        if artist_name:
                            try:
                                tracks = await spotify_service.search_tracks(
                                    f"artist:{artist_name}", limit=3
                                )
                                for track in tracks:
                                    track_artists = ", ".join(
                                        [a["name"] for a in track.get("artists", [])]
                                    )
                                    result += (
                                        f"• **{track['name']}** by {track_artists}\n"
                                    )
                                    result += f"  🎧 [Listen]({track['external_urls']['spotify']})\n\n"
                            except Exception as e:
                                logger.warning(
                                    f"Failed to get tracks for {artist_name}: {e}"
                                )
            else:
                result += f"No similar artists found for '{artist}'. Try checking the spelling or try a different artist."

            return [TextContent(type="text", text=result)]

        except Exception as e:
            logger.error(f"Error finding similar music: {e}")
            return [
                TextContent(
                    type="text", text=f"❌ Error finding similar music: {str(e)}"
                )
            ]

    elif name == "discover_by_mood":
        try:
            mood_or_genre = arguments["mood_or_genre"]
            limit = arguments.get("limit", 20)

            # Get tracks from Last.fm by tag
            lastfm_tracks = await lastfm_service.get_top_tracks_by_tag(
                mood_or_genre, limit
            )

            result = f"# 🌈 Discover Music: {mood_or_genre.title()}\n\n"

            if lastfm_tracks:
                # Convert Last.fm tracks to Spotify searches
                spotify_tracks = []

                for lastfm_track in lastfm_tracks[:limit]:
                    artist_name = (
                        lastfm_track.get("artist", {}).get("name", "")
                        if isinstance(lastfm_track.get("artist"), dict)
                        else lastfm_track.get("artist", "")
                    )
                    track_name = lastfm_track.get("name", "")

                    if artist_name and track_name:
                        try:
                            search_query = f"{track_name} {artist_name}"
                            spotify_results = await spotify_service.search_tracks(
                                search_query, limit=1
                            )

                            if spotify_results:
                                spotify_tracks.append(spotify_results[0])

                        except Exception as e:
                            logger.warning(
                                f"Failed to find '{track_name}' by '{artist_name}' on Spotify: {e}"
                            )

                # Display results
                if spotify_tracks:
                    for i, track in enumerate(spotify_tracks, 1):
                        artists = ", ".join(
                            [artist["name"] for artist in track.get("artists", [])]
                        )
                        result += f"{i:2d}. **{track['name']}** by {artists}\n"
                        result += (
                            f"    🎧 [Listen]({track['external_urls']['spotify']})\n\n"
                        )
                else:
                    result += f"Found {len(lastfm_tracks)} tracks in Last.fm but couldn't match them to Spotify. Try a different mood/genre."
            else:
                result += f"No tracks found for mood/genre '{mood_or_genre}'. Try terms like: chill, happy, electronic, rock, indie, jazz"

            return [TextContent(type="text", text=result)]

        except Exception as e:
            logger.error(f"Error discovering by mood: {e}")
            return [
                TextContent(type="text", text=f"❌ Error discovering music: {str(e)}")
            ]

    else:
        raise ValueError(f"Unknown tool: {name}")


async def start_mcp_server() -> None:
    """Start the MCP server"""
    from mcp.server.stdio import stdio_server

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="setlist_selector",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


def main() -> int:
    """Main entry point for the setlist selector MCP server"""
    try:
        logger.info("🎵 Setlist Selector MCP Server starting...")

        # Check for required environment variables
        required_env_vars = [
            "ANTHROPIC_API_KEY",
            "SPOTIFY_CLIENT_ID",
            "SPOTIFY_CLIENT_SECRET",
        ]
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]

        if missing_vars:
            logger.error(
                f"Missing required environment variables: {', '.join(missing_vars)}"
            )
            logger.error("Please set these in your .env file")
            return 1

        # Start the MCP server
        asyncio.run(start_mcp_server())

        return 0  # Success

    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        return 0
    except Exception as e:
        logger.error(f"Server error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
