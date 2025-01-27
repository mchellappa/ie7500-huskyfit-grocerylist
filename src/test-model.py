import spacy

nlp_ner = spacy.load("./output/model-best")
text = '''Portuguese Kale Soup

Julia Kaiser & Muthu Chellappa

½ pound andouille or jalapeno chicken sausages

1 tablespoon olive oil

1 medium onion, chopped

3 carrots, peeled and sliced

3 garlic cloves, minced

3 medium potatoes, chopped

2 cups butternut squash (or sweet potato), peeled and chopped

4 cups chicken stock

2½ cups of water

1 bunch of kale (at least 4 cups) stems removed & coarsely chopped

1 14.5 ounce can red kidney beans

pinch crushed red pepper flakes

salt to taste

Heat a large soup pan with oil spray. Add the sausages and cook for 5 minutes or so to brown, stirring occasionally. Remove the sausages and set aside to cool. When they have cooled, slice the sausage in half lengthwise, then cut into semi-circles 1/2 inch thick. Set aside.

Heat the olive oil on medium in the soup pan, then add the onions. Sauté for 5 minutes, then add carrots and sauté a few more minutes.

Stir in potatoes, then the butternut squash and garlic. Cook for a minute or two.

After the potatoes have been cooking for a couple of minutes, add the chicken stock and water. Bring to a boil, then reduce to simmer on medium for 15 minutes.

When the veggies start to get soft, use a sturdy spoon to smash some of the potato and butternut against the side of the pot to thicken the soup. Stir in the kale in a few batches, allowing it to cook down a bit to create more room. Add salt to taste and a pinch of red pepper flakes.

Stir in the sausage and kidney beans. Cook for another 5 to 10 minutes to let all the flavors meld. More cooking time is great, too, if you've got the time! When it's cooked to your liking, serve with a crusty bread and glass of Portuguese wine to round things out.

'''

doc = nlp_ner(text)
print(f'''Expected
Entity: 1 c., Label: QUANTITY
Entity: peanut butter, Label: INGREDIENT
Entity: 3/4 c., Label: QUANTITY
Entity: graham cracker crumbs, Label: INGREDIENT
Entity: 1 c., Label: QUANTITY
Entity: butter, Label: INGREDIENT
Entity: 1 lb, Label: QUANTITY
Entity: 3 1/2 c., Label: QUANTITY
Entity: powdered sugar, Label: INGREDIENT
Entity: 1 large pkg, Label: QUANTITY
Entity: chocolate chips, Label: INGREDIENT          
          ''')
print(f'''Actual''')
for ent in doc.ents:    
    print(f"Entity: {ent.text}, Label: {ent.label_}")