"""
Setlist Selector - AI-powered music curation MCP server

An intelligent MCP (Model Context Protocol) server that creates personalized
setlists and playlists by integrating Spotify, Last.fm, and AI-powered music curation.

Usage:
    from setlist_selector import SetlistAgent, SpotifyService, LastFmService

    # Create agent
    agent = SetlistAgent()

    # Generate setlist
    setlist = await agent.curate_setlist("high energy rock opener")
"""

__version__ = "0.1.0"
__author__ = "Anthony Riley"
__email__ = "anthonyriley@fastmail.com"

__all__ = ["__version__"]
