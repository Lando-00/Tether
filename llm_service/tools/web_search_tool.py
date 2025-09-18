import os
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime
from newsapi import NewsApiClient
from newsapi.newsapi_exception import NewsAPIException
from dotenv import load_dotenv
from . import register_tool

load_dotenv()
logger = logging.getLogger(__name__)

def _get_client() -> NewsApiClient:
    api_key = os.getenv("NEWSAPI_KEY")
    if not api_key:
        raise ValueError("Environment variable NEWSAPI_KEY not set")
    if len(api_key) != 32:
        raise ValueError("NEWSAPI_KEY must be a 32-character UUID string")
    return NewsApiClient(api_key=api_key)

def _format_article(article: Dict[str, Any]) -> str:
    title = (article.get("title") or "No title").strip()
    src   = (article.get("source", {}).get("name") or "Unknown source").strip()
    desc  = (article.get("description") or "").strip()
    url   = (article.get("url") or "").strip()
    pub   = article.get("publishedAt", "")
    date_str = ""
    if pub:
        try:
            date_str = pub.split("T")[0]
            datetime.strptime(date_str, "%Y-%m-%d")
        except Exception:
            logger.warning(f"Invalid date: {pub}")

    text = title
    if src:
        text += f" [{src}]"
    if desc:
        if len(desc) > 100:
            desc = desc[:97] + "..."
        text += f": {desc}"
    if date_str:
        text += f" ({date_str})"
    if url:
        text += f" - {url}"
    return text

def _validate_common(page_size: Optional[int], page: Optional[int]) -> None:
    if page_size is not None and (not isinstance(page_size, int) or not (1 <= page_size <= 100)):
        raise ValueError("page_size must be an int between 1 and 100")
    if page is not None and (not isinstance(page, int) or page < 1):
        raise ValueError("page must be an int > 0")

@register_tool
def web_search(
    query: str,
    count: int = 5,
    sources: Optional[str] = None,
    domains: Optional[str] = None,
    exclude_domains: Optional[str] = None,
    from_param: Optional[str] = None,
    to: Optional[str] = None,
    language: str = "en",
    sort_by: str = "relevancy",
    page: int = 1
) -> List[str]:
    """
    Search historical news (/v2/everything).

    Args:
      query (str): Required.
      count (int): 1–100 (default 5).
      sources, domains, exclude_domains (str): comma-separated.
      from_param, to (str): YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS.
      language (str): 2-letter ISO code.
      sort_by (str): 'relevancy', 'popularity', 'publishedAt'.
      page (int): >0.

    Returns:
      List[str]: Formatted article strings or error list.
    """
    query = query.strip()
    if not query:
        return ["Error: query must be a non-empty string"]

    try:
        _validate_common(count, page)
        if sort_by not in {"relevancy","popularity","publishedAt"}:
            return [f"Error: sort_by must be one of ['relevancy','popularity','publishedAt']"]
        if len(language) != 2:
            return ["Error: language must be a 2-letter ISO code"]

        date_args: Dict[str,str] = {}
        for name, val in (("from_param",from_param),("to",to)):
            if val:
                try:
                    _ = datetime.fromisoformat(val.replace("Z","")) if "T" in val else datetime.strptime(val,"%Y-%m-%d")
                except Exception:
                    return [f"Error: {name} must be YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS"]
                date_args[name] = val

        client = _get_client()
        resp = client.get_everything(
            q=query, page_size=count, language=language, sort_by=sort_by,
            page=page, **date_args,
            sources=sources or None,
            domains=domains or None,
            exclude_domains=exclude_domains or None
        )
    except NewsAPIException as e:
        return [f"NewsAPI error: {e}"]
    except ValueError as e:
        return [f"Parameter error: {e}"]
    except Exception as e:
        logger.error(f"web_search error: {e}")
        return [f"Unexpected error: {e}"]

    if resp.get("status") != "ok":
        return [f"NewsAPI error ({resp.get('code')}): {resp.get('message')}"]

    articles = resp.get("articles", [])
    if not articles:
        return [f"No articles for query='{query}'"]

    return [_format_article(a) for a in articles[:count]]

@register_tool
def get_top_headlines(
    q: Optional[str] = None,
    qintitle: Optional[str] = None,
    sources: Optional[str] = None,
    language: str = "en",
    country: Optional[str] = None,
    category: Optional[str] = None,
    page_size: int = 5,
    page: int = 1
) -> List[str]:
    """
    Fetch live top headlines (/v2/top-headlines).

    Args:
      q, qintitle, sources (str): comma-separated.
      language (str): default 'en'.
      country, category (str): cannot mix with sources.
      page_size (int): 1–100.
      page (int): >0.

    Returns:
      List[str]: Formatted headline strings or error list.
    """
    try:
        _validate_common(page_size, page)
        if sources and (country or category):
            return ["Error: cannot mix sources with country/category"]
        valid_cats = {"business","entertainment","general","health","science","sports","technology"}
        if category and category not in valid_cats:
            return [f"Error: category must be one of {sorted(valid_cats)}"]
        if len(language) != 2:
            return ["Error: language must be 2-letter ISO code"]

        params = {
            **({"q":q.strip()} if q else {}),
            **({"qintitle":qintitle.strip()} if qintitle else {}),
            "language":language,
            "page_size":page_size,
            "page":page,
            **({"sources":sources} if sources else {}),
            **({"country":country} if country else {}),
            **({"category":category} if category else {}),
        }

        resp = _get_client().get_top_headlines(**params)
    except NewsAPIException as e:
        return [f"NewsAPI error: {e}"]
    except ValueError as e:
        return [f"Parameter error: {e}"]
    except Exception as e:
        logger.error(f"get_top_headlines error: {e}")
        return [f"Unexpected error: {e}"]

    if resp.get("status") != "ok":
        return [f"NewsAPI error ({resp.get('code')}): {resp.get('message')}"]

    articles = resp.get("articles", [])
    if not articles:
        return ["No top headlines found."]
    return [_format_article(a) for a in articles[:page_size]]

@register_tool
def list_sources(
    category: Optional[str] = None,
    language: Optional[str] = None,
    country: Optional[str] = None
) -> List[str]:
    """
    List sources (/v2/sources).

    Args:
      category, language, country (str): filters.

    Returns:
      List[str]: Formatted source info or error list.
    """
    try:
        valid_cats = {"business","entertainment","general","health","science","sports","technology"}
        params: Dict[str,str] = {}
        if category:
            if category not in valid_cats:
                return [f"Error: category must be one of {sorted(valid_cats)}"]
            params["category"] = category
        if language and len(language)==2:
            params["language"] = language.lower()
        elif language:
            return ["Error: language must be 2-letter ISO code"]
        if country and len(country)==2:
            params["country"] = country.lower()
        elif country:
            return ["Error: country must be 2-letter ISO code"]

        resp = _get_client().get_sources(**params)
    except NewsAPIException as e:
        return [f"NewsAPI error: {e}"]
    except Exception as e:
        logger.error(f"list_sources error: {e}")
        return [f"Unexpected error: {e}"]

    if resp.get("status") != "ok":
        return [f"NewsAPI error ({resp.get('code')}): {resp.get('message')}"]

    sources = resp.get("sources", [])
    if not sources:
        return ["No sources found."]
    out: List[str] = []
    for src in sources:
        try:
            name = src.get("name","Unknown")
            desc = src.get("description","")
            url = src.get("url","")
            md = []
            for k in ("category","language","country"):
                v=src.get(k)
                if v: md.append(str(v).upper())
            line=name
            if desc:
                line+=f": {desc[:77]+'...' if len(desc)>80 else desc}"
            if md:
                line+=f" ({', '.join(md)})"
            if url:
                line+=f" - {url}"
            out.append(line)
        except:
            out.append(src.get("name","Unknown"))
    return out