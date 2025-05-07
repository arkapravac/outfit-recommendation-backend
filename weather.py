import aiohttp
import os
from datetime import datetime
from typing import Dict, List

class WeatherService:
    def __init__(self):
        self.api_key = os.getenv('OPENWEATHER_API_KEY')
        self.base_url = 'http://api.openweathermap.org/data/2.5'

    async def get_weather(self, city: str) -> Dict:
        async with aiohttp.ClientSession() as session:
            url = f"{self.base_url}/weather?q={city}&appid={self.api_key}&units=metric"
            async with session.get(url) as response:
                return await response.json()

    def get_weather_based_recommendations(self, weather_data: Dict, products: List[Dict]) -> List[Dict]:
        temp = weather_data['main']['temp']
        weather_condition = weather_data['weather'][0]['main'].lower()
        
        # Define weather-based rules
        weather_rules = {
            'rain': ['raincoat', 'umbrella', 'boots'],
            'snow': ['winter coat', 'boots', 'scarf'],
            'clear': ['sunglasses', 'hat'] if temp > 25 else ['light jacket'],
            'clouds': ['light jacket'] if temp < 20 else ['casual wear']
        }

        # Get suitable categories for current weather
        suitable_categories = weather_rules.get(weather_condition, ['casual wear'])
        
        # Filter products based on weather conditions
        recommended_products = []
        for product in products:
            if any(category.lower() in product['category'].lower() for category in suitable_categories):
                # Add weather suitability score
                product['weather_score'] = self._calculate_weather_score(
                    product, temp, weather_condition)
                recommended_products.append(product)
        
        # Sort by weather suitability score
        return sorted(recommended_products, key=lambda x: x['weather_score'], reverse=True)

    def _calculate_weather_score(self, product: Dict, temp: float, weather: str) -> float:
        score = 1.0
        
        # Temperature-based scoring
        if 'season' in product:
            if temp < 10 and 'winter' in product['season']:
                score *= 1.5
            elif temp > 25 and 'summer' in product['season']:
                score *= 1.5
            elif 10 <= temp <= 25 and 'spring' in product['season']:
                score *= 1.3
        
        # Weather condition scoring
        if weather == 'rain' and 'waterproof' in product.get('attributes', {}):
            score *= 1.4
        elif weather == 'clear' and 'sun_protection' in product.get('attributes', {}):
            score *= 1.3
        
        return score