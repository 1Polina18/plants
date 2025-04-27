import os
import asyncio
import aiohttp
from bs4 import BeautifulSoup

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

async def download_image_async(session, url, folder_name, image_name):
    try:
        async with session.get(url) as response:
            if response.status == 200:
                content = await response.read()
                with open(os.path.join(folder_name, image_name), 'wb') as f:
                    f.write(content)
                print(f"Скачано: {image_name}")
            else:
                print(f"Не удалось получить изображение с {url}: {response.status}")
    except Exception as e:
        print(f"Ошибка при скачивании {url}: {e}")
    await asyncio.sleep(0.5)


async def scrape_images_async(plant_name, num_images=10):
    folder_name = os.path.join('data', plant_name.replace(" ", "_"))
    create_directory(folder_name)

    images_downloaded = 0
    page_number = 0

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }

    async with aiohttp.ClientSession(headers=headers) as session:
        while images_downloaded < num_images:
            search_url = f"https://www.google.com/search?hl=en&tbm=isch&q={plant_name}&start={page_number * 20}"

            try:
                async with session.get(search_url) as response:
                    if response.status != 200:
                        print("Не удалось получить результаты поиска")
                        break
                    soup = BeautifulSoup(await response.text(), 'html.parser')

                    images = soup.find_all('img')[1:]

                    tasks = []
                    for i, img in enumerate(images):
                       if images_downloaded >= num_images:
                           break

                       try:
                           img_url = img['src']
                           tasks.append(download_image_async(session, img_url, folder_name, f"{plant_name}_{images_downloaded + 1}.jpg"))
                           images_downloaded += 1
                       except KeyError:
                           print(f"У изображения {i} нет атрибута 'src'. Оно будет пропущено.")
                           continue

                    await asyncio.gather(*tasks)
                    page_number += 1

            except aiohttp.ClientError as e:
                print(f"Ошибка aiohttp: {e}")
                await asyncio.sleep(5)

            except Exception as e:
                print(f"Произошла непредвиденная ошибка: {e}")
                break


if __name__ == "__main__":
    plant_name = input("Введите название растения для поиска изображений: ")
    num_images = int(input("Введите количество изображений для загрузки: "))
    asyncio.run(scrape_images_async(plant_name, num_images))
