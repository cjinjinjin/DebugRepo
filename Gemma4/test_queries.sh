#!/bin/bash
# Gemma4 Two-Step T2I Prompt Generation - Test Queries
# Usage: bash test_queries.sh [port]
# Default port: 8899

PORT=${1:-8899}
URL="http://localhost:$PORT"

echo "=========================================="
echo "Testing Gemma4 T2I on port $PORT"
echo "=========================================="

# Case 1: Ford F-150 Lightning Dealership
echo ""
echo ">>> Case 1: Ford F-150 Lightning Dealership"
echo "------------------------------------------"
curl -s -X POST "$URL" \
  -H "Content-Type: application/json" \
  -d '{
    "landing_page_content": "Vehicles Shop Support & Service Ford Pro Solutions Locate Dealer Pricing for and Offers Home Incentives & Offers Competitive Compare Trade-In Value Locate a Dealer Mustang® Escape® Bronco Sport® Bronco® Explorer® Mustang Mach-E® Expedition® Maverick® Ranger® F-150® Super Duty® E-Transit™ F-150 Lightning Transit® VanWagon Shop at Your Paramus, Bridgeport and White Plains Local Ford Dealers. View current cash incentives, APR financing and lease offers for a new car, truck, suv at your local Ford Dealership.",
    "url": "https://www.ford.com/local/paramus-bridgeport-white-plains/incentives-offers/?model=F-150+Lightning",
    "num_prompts": 5
  }' | python3 -m json.tool
echo ""

# Case 2: HSN Women's Clothing
echo ""
echo ">>> Case 2: HSN Women's Clothing"
echo "------------------------------------------"
curl -s -X POST "$URL" \
  -H "Content-Type: application/json" \
  -d '{
    "landing_page_content": "Save $15 on your first order with code WELCOME2025* Fashion Women'\''s Clothing Pants Dresses Tops Sweaters Jackets & outerwear Jeans. Nina Leonard 3-piece Textured Knit Cardigan, Tank & Pant Set. Comfort Code Silky Velvet Satin Trim 2-piece Top and Pant Sleep Set. G by Giuliana Jacquard Knit Sweater with Sequins. G by Giuliana Rhinestone Collar Pullover Sweater. G by Giuliana Black Label Knit Teddy Luxe Coat. Upgrade your wardrobe with women'\''s apparel from HSN. Shop this amazing collection of women'\''s clothing and find a new garment to fit your individual style.",
    "url": "https://www.hsn.com/shop/womens-clothing/fa0153",
    "num_prompts": 5
  }' | python3 -m json.tool
echo ""

# Case 3: OpenRent Property Letting
echo ""
echo ">>> Case 3: OpenRent Property Letting"
echo "------------------------------------------"
curl -s -X POST "$URL" \
  -H "Content-Type: application/json" \
  -d '{
    "landing_page_content": "No hidden fees. You do the viewings, we do the rest. Advertise Now for Free Other Landlord Services Tenants pay no admin fees. Free Add Listing Now For Free 4 Months Advertising. Landlord Services: Viewing and Enquiry Organiser. Professional Tenancy Creation: Contract drafting and digital signing. Deposit Registration. Initial Rent Collection. Industry Leading Referencing. We advertise on Rightmove, OnTheMarket and Many More. Discover the cheapest way to let your property.",
    "url": "https://www.openrent.co.uk/our-pricing",
    "num_prompts": 5
  }' | python3 -m json.tool
echo ""

# Case 4: Instant Gaming
echo ""
echo ">>> Case 4: Instant Gaming"
echo "------------------------------------------"
curl -s -X POST "$URL" \
  -H "Content-Type: application/json" \
  -d '{
    "landing_page_content": "Trending ARC Raiders PC Steam. Quarantine Zone The Last Check PC Steam. Cult of the Lamb Woolhaven PC Steam. StarRupture PC Steam. Reliable and safe Over 20000 games Customer support Human support 24/7. Pre-orders The Sims 4 Royalty Legacy Grand Bundle. Dragon Quest VII Reimagined Digital Deluxe Edition. Crimson Desert Deluxe Edition. Instant Gaming is an amazing platform to buy your PC PlayStation Xbox and Switch games cheaper with a 24/7 instant delivery. Buy your video games cheaper for PC and consoles with the best deals offers promotions and discounts.",
    "url": "https://www.instant-gaming.com/en/",
    "num_prompts": 5
  }' | python3 -m json.tool
echo ""

echo "=========================================="
echo "All cases completed"
echo "=========================================="
