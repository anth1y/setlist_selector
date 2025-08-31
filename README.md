# Setlist Selector

An intelligent MCP (Model Context Protocol) server that creates personalized setlists and playlists by integrating Spotify, Last.fm, and AI-powered music curation.

## Features

🎵 **AI-Powered Curation**: Uses Claude (Anthropic) to understand natural language setlist requests and create intelligent music selections
🎧 **Multi-Service Integration**: Combines Spotify's vast catalog with Last.fm's music intelligence
⚡ **MCP Compatible**: Works seamlessly with ChatGPT, Claude, and other MCP-supporting AI tools
🎨 **Creative Setlist Generation**: Creates themed setlists from simple descriptions like "high-energy rock show opener" or "intimate acoustic coffee shop set"
🔍 **Music Discovery**: Find similar artists, explore by mood/genre, and discover new music based on your preferences

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone git@github.com:anth1y/setlist-selector.git
cd setlist-selector

# Install uv
https://docs.astral.sh/uv/getting-started/installation/

# Setup venv
uv venv

# Activate venv
source .venv/bin/activate

# Install dependencies using uv
uv sync
```

### 2. API Setup

You'll need API keys from three services:

#### Spotify API
1. Go to https://developer.spotify.com/dashboard/applications
2. Create a new app
3. Note your Client ID and Client Secret

#### Last.fm API
1. Create account at https://www.last.fm
2. Get API key at https://www.last.fm/api/account/create

#### Anthropic API
1. Get API key from https://console.anthropic.com

### 3. Environment Setup

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and insert your API keys
vim .env
```

### 4. Run the MCP Server

```bash
# Using uv
uv run python -m setlist_selector.main

# Or with regular python
python -m setlist_selector.main
```

### 5. Connect to AI Interface

Add this server to your MCP-compatible AI interface:

**For Claude Desktop:**
```json
{
  "mcpServers": {
    "setlist-selector": {
      "command": "setlist_selector/start_mcp.sh",
      "args": []
    }
  }
}
```

Be sure to restart your claude desktop app after adding the above configuration.

## Available Tools

### 1. `create_ai_playlist`
Creates an intelligent setlist/playlist based on natural language descriptions using AI curation.

**Parameters:**
- `prompt` (required): Describe your desired setlist (e.g., "upbeat workout music", "chill Sunday morning vibes", "high-energy rock concert opener")
- `include_recommendations`: Whether to include Spotify's AI recommendations (default: true)

**Example Usage:**
```
Create a setlist for "rainy day indie music with acoustic vibes"
```

### 2. `search_songs`
Search for specific songs or artists on Spotify.

**Parameters:**
- `query` (required): Search term (artist, song, or keywords)
- `limit`: Number of results (1-50, default: 10)

### 3. `get_similar_music`
Discover similar artists and tracks based on a given artist using Last.fm's music intelligence.

**Parameters:**
- `artist` (required): Artist name to find similar music for
- `include_tracks`: Whether to include track recommendations (default: true)

### 4. `discover_by_mood`
Explore music by mood, genre, or vibe using Last.fm's extensive tag system.

**Parameters:**
- `mood_or_genre` (required): Mood or genre tag (e.g., 'happy', 'chill', 'electronic', 'indie')
- `limit`: Number of tracks to discover (default: 20)

## Architecture

```
AI Model (ChatGPT/Claude)
    ↓ MCP Protocol
Your MCP Server
    ↓ Anthropic API
Agent Logic (Claude Music Curation)
    ↓ REST APIs
Music Services (Spotify + Last.fm)
```

## Example Workflows

### Creative Setlist Generation
- **Input**: "Create a setlist for a cozy coffee shop atmosphere"
- **AI Processing**: Analyzes the mood, suggests artists and genres
- **Output**: Curated setlist with indie folk, acoustic, and ambient tracks

### Music Discovery
- **Input**: "I love Bon Iver, find me similar artists"
- **Processing**: Uses Last.fm similarity data + Spotify recommendations
- **Output**: Artists like Phoebe Bridgers, Fleet Foxes, The National with sample tracks

### Mood-Based Exploration
- **Input**: "Discover some energetic electronic music"
- **Processing**: Searches Last.fm tags + Spotify's catalog
- **Output**: High-energy electronic tracks across subgenres

## Testing

### Testing Tools
```bash
# Install test dependencies
uv sync --group test

# Run tests
uv run pytest tests/ -v

# Test with real APIs (requires valid API keys)
uv run pytest tests/integration/ -v --real-apis

# Run with coverage
uv run pytest --cov=setlist_selector --cov-report=html
```

### Failure Scenarios
- **API Rate Limits**: Implement exponential backoff and retry logic
- **Invalid Search Results**: Fallback to generic recommendations
- **Network Failures**: Graceful degradation with cached results
- **Malformed AI Responses**: Structured fallback setlists

## Security Considerations

### Current Protections
- Environment variable storage for sensitive API keys
- Input validation and sanitization for all search queries
- Rate limiting awareness to prevent API abuse
- No storage of user authentication tokens (uses client credentials flow)

### Potential Vulnerabilities & Mitigations
- **API Key Exposure**: Store in secure environment variables, rotate keys regularly
- **Injection Attacks**: Sanitize all search queries and setlist names, validate AI outputs
- **Rate Limit Abuse**: Implement request queuing and throttling, monitor usage patterns
- **Data Privacy**: No persistent storage of user music preferences, clear API request logging

### Production Recommendations
- Use OAuth 2.0 PKCE flow for user authentication in production
- Implement request signing for additional API security
- Add IP whitelisting and CORS policies
- Regular dependency security audits

## Performance Considerations

### Current Performance
- **Response Time**: 3-8 seconds per setlist creation (depends on Claude + API latency)
- **Throughput**: Limited by Spotify API (100 requests/minute) and Anthropic rate limits
- **Concurrent Users**: Single-threaded, processes one request at a time
- **Setlist Size**: Optimized for 10-50 track setlists

### Scaling Strategies
- **Horizontal Scaling**: Deploy multiple server instances with load balancer
- **Caching**: Cache Spotify search results, Last.fm artist similarity data, AI responses
- **Async Processing**: Queue setlist generation for background processing
- **Database Integration**: Store user preferences and setlist history

### Performance Bottlenecks
- Claude API latency (1-3 seconds for setlist curation)
- Spotify API rate limits (100 requests/minute for search)
- Last.fm API rate limits (varies, generally permissive)
- Network latency for multiple API calls per setlist

## Caveats & Gotchas

⚠️ **API Dependencies**: Requires active internet connection and valid API keys for all services
⚠️ **Rate Limits**: Spotify (100/min), Last.fm (varies), Anthropic (tier-dependent) - may cause delays
⚠️ **Music Availability**: Song availability varies by region due to Spotify licensing
⚠️ **AI Variability**: AI-generated setlists may have different quality/relevance across requests
⚠️ **API Costs**: Anthropic charges per token (~$0.01-0.02 per setlist), Spotify/Last.fm are free with limits
⚠️ **Authentication Scope**: Current implementation uses client credentials (public data only)

### Caveats & Gotchas
⚠️ **API Dependencies**: Requires active internet connection and valid API keys for all services
⚠️ **Rate Limits**: Spotify (100/min), Last.fm (varies), Anthropic (tier-dependent) - may cause delays
⚠️ **Music Availability**: Song availability varies by region due to Spotify licensing
⚠️ **AI Variability**: AI-generated setlists may have different quality/relevance across requests
⚠️ **API Costs**: Anthropic charges per token (~$0.01-0.02 per setlist), Spotify/Last.fm are free with limits
⚠️ **Authentication Scope**: Current implementation uses client credentials (public data only)

### Important Notes
- **No Setlist Saving**: This version searches and recommends but doesn't save setlists to Spotify accounts
- **Regional Restrictions**: Some tracks may not be available in all countries
- **Real-time Data**: Music popularity and availability can change
- **API Versioning**: External APIs may change, requiring updates

## Future Improvements

### With More Time (Priority Order)

1.Enhanced Music Intelligence
   * Audio feature analysis for better matching (tempo, key, energy)
   * Machine learning models for personalized recommendations
   * Integration with more music services (Apple Music, YouTube Music)

2. User Experience Features
   * Setlist saving to Spotify accounts (requires user OAuth)
   * Web interface for setlist management and sharing
   * Real-time collaborative setlist creation

3. Advanced AI Capabilities
   * Contextual awareness (time of day, weather, calendar events)
   * Learning from user feedback and preferences
   * Multi-language setlist descriptions and international music discovery

4. Performance & Reliability
   * Redis caching for frequently accessed data
   * Background job processing for large setlists
   * Comprehensive monitoring and alerting
   * A/B testing for setlist quality optimization

5. Enterprise Features
   * Multi-user support with individual preferences
   * Analytics dashboard for music trends and usage
   * API rate limit management and scaling
   * Integration with business music platforms

 ### Expanded Functionality
   * Mood Detection: Analyze user's recent music to suggest mood-appropriate setlists
   * Event-Based Setlists: Generate music for specific events, seasons, or activities
   * Social Features: Share and collaborate on setlists with friends
   * Cross-Platform Sync: Sync setlists across multiple music services
   * Voice Integration: Voice commands for setlist creation and control
   * Visual Analysis: Generate setlists from image mood boards or color palettes

### Development Setup
```bash
# Install development dependencies
uv sync --group dev

# Set up pre-commit hooks
uv run pre-commit install

# Run linting and formatting
uv run black . && uv run flake8 . && uv run mypy .

# Run tests in watch mode
uv run pytest --watch
```

## License

MIT License - see [LICENSE](LICENSE) file for details

---

## API Reference

### Music Service APIs Used
- **Spotify Web API**: Track search, recommendations, audio features
- **Last.fm API**: Artist similarity, music tags, genre exploration
- **Anthropic API**: Natural language setlist curation and music understanding

### Rate Limits Summary
- Spotify: 100 requests/minute (search), 1000/hour (recommendations)
- Last.fm: ~5 requests/second (varies by method)
- Anthropic: Depends on plan (typically 4000 requests/minute for Claude 3.5 Haiku)

Built with ❤️ for music lovers and AI enthusiasts
