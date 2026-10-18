import os
import logging
import requests

logger = logging.getLogger(__name__)

class TrichyWeatherClassifier:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("OPENWEATHER_API_KEY", "")
        self.base_url = "https://api.openweathermap.org/data/2.5/weather"
        # Trichy's approximate coordinates (lat/lon)
        self.trichy_lat = 10.7905
        self.trichy_lon = 78.7047

    def classify_weather(self):
        if not self.api_key:
            logger.warning("No OpenWeatherMap API key configured. Falling back to 'average' weather.")
            return "average"

        try:
            params = {
                "lat": self.trichy_lat,
                "lon": self.trichy_lon,
                "appid": self.api_key,
                "units": "metric"  # Temperature in Celsius
            }

            response = requests.get(self.base_url, params=params, timeout=5)
            response.raise_for_status()

            weather_data = response.json()
            return self._evaluate_weather(weather_data)

        except requests.exceptions.RequestException as e:
            logger.warning("Weather API request failed (%s). Defaulting to 'average' weather.", type(e).__name__)
            return "average"
        except (KeyError, ValueError, TypeError) as e:
            logger.warning("Failed to parse weather data (%s). Defaulting to 'average' weather.", type(e).__name__)
            return "average"

    def _evaluate_weather(self, data):
        try:
            main_data = data.get("main", {})
            temp = float(main_data.get("temp", 25.0))
            humidity = float(main_data.get("humidity", 50.0))
            rain_data = data.get("rain", {})
            rain = float(rain_data.get("1h", 0)) if isinstance(rain_data, dict) else 0.0
            wind_data = data.get("wind", {})
            wind_speed = float(wind_data.get("speed", 3.0)) if isinstance(wind_data, dict) else 0.0

            # Classification Logic
            if (temp > 36 or temp < 18) or (humidity > 85) or (rain > 15) or (wind_speed > 10):
                return "bad"  # Extreme heat/cold, heavy rain, or strong winds
            elif (20 <= temp <= 30) and (humidity < 70) and (rain < 2.5):
                return "good"  # Pleasant conditions
            else:
                return "average"  # Moderate weather
        except Exception as e:
            logger.warning("Error evaluating weather metrics (%s). Defaulting to 'average'.", type(e).__name__)
            return "average"

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    api_key = os.environ.get("OPENWEATHER_API_KEY", "")
    classifier = TrichyWeatherClassifier(api_key)
    result = classifier.classify_weather()
    print(f"Current weather classification in Trichy: {result.upper()}")