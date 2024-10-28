#!/usr/bin/env python3
import nbformat as nbf

def create_simplified_notebook():
    nb = nbf.v4.new_notebook()
    
    # Introduction
    nb.cells.append(nbf.v4.new_markdown_cell("""
# Social Media Marketing Automation
## A Practical Guide to LinkedIn and Instagram Automation

This notebook provides a simple way to:
1. Generate optimized social media content
2. Post automatically to LinkedIn and Instagram
3. Track post performance
4. Get AI-powered improvement suggestions

### Initial Setup

First, you'll need:
1. LinkedIn Developer Account - [Get Here](https://www.linkedin.com/developers/)
2. Instagram Business Account - [Setup Guide](https://business.instagram.com/)
3. OpenAI API Key - [Get Here](https://platform.openai.com/)
"""))

    # Setup Section
    nb.cells.append(nbf.v4.new_markdown_cell("""
## 1. Setup

Run this cell to install required packages:
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
!pip install swarms python-dotenv linkedin-api facebook-business pandas
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""
### Configure Your API Keys

Create a file named `.env` in the same directory as this notebook with your API keys:
```
LINKEDIN_ACCESS_TOKEN=your_token_here
INSTAGRAM_ACCESS_TOKEN=your_token_here
OPENAI_API_KEY=your_key_here
```
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
import os
from dotenv import load_dotenv
from swarms import Agent
from swarm_models import OpenAIChat
import json
from datetime import datetime

# Load your API keys
load_dotenv()

# Initialize OpenAI
model = OpenAIChat(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model_name="gpt-4",
    temperature=0.7
)

# Initialize API connections
def initialize_social_apis():
    try:
        from linkedin import linkedin
        from facebook_business.api import FacebookAdsApi
        
        # LinkedIn setup
        linkedin_api = linkedin.LinkedInApplication(
            token=os.getenv("LINKEDIN_ACCESS_TOKEN")
        )
        
        # Instagram setup
        FacebookAdsApi.init(
            access_token=os.getenv("INSTAGRAM_ACCESS_TOKEN")
        )
        
        return {
            'linkedin': linkedin_api,
            'instagram': FacebookAdsApi.get_default_api()
        }
    except Exception as e:
        print(f"Error setting up APIs: {str(e)}")
        print("Please check your API keys and try again.")
        return None

apis = initialize_social_apis()
"""))

    # Content Creation
    nb.cells.append(nbf.v4.new_markdown_cell("""
## 2. Content Creation

Let's create some social media content! Just fill in your content brief below.
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
# Easy-to-use content generation function
def generate_social_content(
    topic: str,
    key_points: list,
    tone: str = "professional",
    target_audience: str = "professionals"
):
    \"\"\"
    Generate social media content based on your inputs
    
    Args:
        topic: Main topic or subject
        key_points: List of main points to cover
        tone: Desired tone (professional, casual, excited, etc.)
        target_audience: Who you're targeting
    
    Returns:
        Dictionary with content for each platform
    \"\"\"
    
    content_prompt = \"\"\"
    You are a Social Media Marketing Expert. Create engaging posts for LinkedIn and Instagram.
    
    Topic: {topic}
    Key Points: {points}
    Tone: {tone}
    Target Audience: {audience}
    
    Return ONLY a JSON object with this exact structure:
    {{
        "linkedin": {{
            "text": "main post content",
            "hashtags": ["tag1", "tag2"],
        }},
        "instagram": {{
            "caption": "main caption",
            "hashtags": ["tag1", "tag2"]
        }}
    }}
    \"\"\"
    
    # Create our content generation agent
    content_agent = Agent(
        agent_name="Social-Content-Creator",
        system_prompt=content_prompt.format(
            topic=topic,
            points=key_points,
            tone=tone,
            audience=target_audience
        ),
        llm=model,
        max_loops=1,
        output_type="json"
    )
    
    # Generate content
    try:
        result = json.loads(content_agent.run("Generate social media posts"))
        return result
    except Exception as e:
        return {"error": str(e)}

# Example usage
content = generate_social_content(
    topic="New AI Product Launch",
    key_points=[
        "Revolutionary AI technology",
        "Increases productivity by 50%",
        "Easy to integrate"
    ],
    tone="exciting",
    target_audience="tech professionals"
)

print("Generated Content:")
print(json.dumps(content, indent=2))
"""))

    # Posting Content
    nb.cells.append(nbf.v4.new_markdown_cell("""
## 3. Posting Your Content

Now let's post your content to social media!
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
class SocialMediaPoster:
    def __init__(self, apis):
        self.linkedin_api = apis['linkedin']
        self.instagram_api = apis['instagram']
    
    def post_to_linkedin(self, content):
        \"\"\"Post to LinkedIn\"\"\"
        try:
            # Combine text and hashtags
            full_text = f"{content['text']}\\n\\n{' '.join(content['hashtags'])}"
            
            response = self.linkedin_api.submit_share({
                'comment': full_text,
                'visibility': {
                    'code': 'anyone'
                }
            })
            
            return {
                'success': True,
                'post_id': response.get('id'),
                'platform': 'linkedin',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'platform': 'linkedin'
            }
    
    def post_to_instagram(self, content):
        \"\"\"Post to Instagram\"\"\"
        try:
            # Combine caption and hashtags
            full_caption = f"{content['caption']}\\n\\n{' '.join(content['hashtags'])}"
            
            response = self.instagram_api.create_post({
                'caption': full_caption,
                'media_type': 'CAROUSEL'
            })
            
            return {
                'success': True,
                'post_id': response.get('id'),
                'platform': 'instagram',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'platform': 'instagram'
            }
    
    def post_content(self, content):
        \"\"\"Post to all platforms\"\"\"
        results = {
            'linkedin': self.post_to_linkedin(content['linkedin']),
            'instagram': self.post_to_instagram(content['instagram'])
        }
        return results

# Create poster instance
poster = SocialMediaPoster(apis)

# Post the content we generated earlier
posting_results = poster.post_content(content)
print("Posting Results:")
print(json.dumps(posting_results, indent=2))
"""))

    # Performance Tracking
    nb.cells.append(nbf.v4.new_markdown_cell("""
## 4. Track Your Post Performance

Let's see how your posts are performing!
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
class PerformanceTracker:
    def __init__(self, apis):
        self.linkedin_api = apis['linkedin']
        self.instagram_api = apis['instagram']
    
    def get_linkedin_metrics(self, post_id):
        \"\"\"Get LinkedIn post performance\"\"\"
        try:
            metrics = self.linkedin_api.get_share_stats(post_id)
            return {
                'impressions': metrics.get('impressionCount', 0),
                'engagement': metrics.get('engagementCount', 0),
                'likes': metrics.get('likeCount', 0),
                'comments': metrics.get('commentCount', 0),
                'shares': metrics.get('shareCount', 0)
            }
        except Exception as e:
            return {'error': str(e)}
    
    def get_instagram_metrics(self, post_id):
        \"\"\"Get Instagram post performance\"\"\"
        try:
            metrics = self.instagram_api.get_insights(
                post_id,
                metrics=['impressions', 'reach', 'engagement']
            )
            return {
                'impressions': metrics.get('impressions', 0),
                'reach': metrics.get('reach', 0),
                'engagement': metrics.get('engagement', 0)
            }
        except Exception as e:
            return {'error': str(e)}
    
    def get_performance(self, post_ids):
        \"\"\"Get performance for all posts\"\"\"
        return {
            'linkedin': self.get_linkedin_metrics(post_ids['linkedin']),
            'instagram': self.get_instagram_metrics(post_ids['instagram'])
        }

# Create tracker instance
tracker = PerformanceTracker(apis)

# Get post IDs from our posting results
post_ids = {
    'linkedin': posting_results['linkedin'].get('post_id'),
    'instagram': posting_results['instagram'].get('post_id')
}

# Track performance
performance = tracker.get_performance(post_ids)
print("Post Performance:")
print(json.dumps(performance, indent=2))
"""))

    # Get Improvements
    nb.cells.append(nbf.v4.new_markdown_cell("""
## 5. Get AI-Powered Improvement Suggestions

Let's analyze your post performance and get suggestions for improvement!
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
def get_improvements(performance_data):
    \"\"\"Get AI-powered suggestions for improvement\"\"\"
    
    improvement_prompt = \"\"\"
    You are a Social Media Optimization Expert. Based on the performance data,
    provide specific, actionable suggestions for improvement.
    
    Return ONLY a JSON object with this exact structure:
    {
        "analysis": {
            "strengths": ["strength1", "strength2"],
            "areas_for_improvement": ["area1", "area2"]
        },
        "suggestions": {
            "content": ["suggestion1", "suggestion2"],
            "timing": ["suggestion1", "suggestion2"],
            "engagement": ["suggestion1", "suggestion2"]
        },
        "next_steps": ["step1", "step2"]
    }
    \"\"\"
    
    optimization_agent = Agent(
        agent_name="Performance-Optimizer",
        system_prompt=improvement_prompt,
        llm=model,
        max_loops=1,
        output_type="json"
    )
    
    try:
        suggestions = json.loads(
            optimization_agent.run(json.dumps(performance_data))
        )
        return suggestions
    except Exception as e:
        return {"error": str(e)}

# Get improvement suggestions
improvements = get_improvements(performance)
print("Improvement Suggestions:")
print(json.dumps(improvements, indent=2))
"""))

    # Quick Start Guide
    nb.cells.append(nbf.v4.new_markdown_cell("""
## Quick Start Guide

1. Set up your environment:
   - Create `.env` file with your API keys
   - Run the setup cells

2. Generate content:
```python
content = generate_social_content(
    topic="Your Topic",
    key_points=["Point 1", "Point 2", "Point 3"],
    tone="professional",
    target_audience="your audience"
)
```

3. Post content:
```python
posting_results = poster.post_content(content)
```

4. Track performance:
```python
performance = tracker.get_performance(post_ids)
```

5. Get improvements:
```python
improvements = get_improvements(performance)
```

## Need Help?

- Check your API keys if you get connection errors
- Make sure your Instagram account is a Business account
- Verify your LinkedIn Developer credentials
- Join our [Discord community](https://discord.gg/kS3rwKs3ZC) for support
"""))

    # Save the notebook
    output_file = "social_media_automation.ipynb"
    with open(output_file, "w") as f:
        nbf.write(nb, f)
    
    return output_file

def main():
    try:
        output_file = create_simplified_notebook()
        print(f"Successfully created notebook: {output_file}")
        print("\nYou can now open this notebook using:")
        print(f"jupyter notebook {output_file}")
    except Exception as e:
        print(f"Error creating notebook: {str(e)}")

if __name__ == "__main__":
    main()