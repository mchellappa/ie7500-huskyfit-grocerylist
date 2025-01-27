export interface RecipeRequest {
    ingredients: string;
}

export interface QuantityEntity {
    start: number;
    end: number;
    label: 'QUANTITY';
}

export interface IngredientEntity {
    start: number;
    end: number;
    label: 'INGREDIENT';
}

export interface ProcessedResponse {
    quantities: QuantityEntity[];
    ingredients: IngredientEntity[];
}