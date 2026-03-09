from typing import List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import pymorphy3
import graphviz
import nltk
from nltk.stem.snowball import SnowballStemmer

try:
    stemmer = SnowballStemmer("russian")
except:
    nltk.download('punkt')
    stemmer = SnowballStemmer("russian")
                              
morph = pymorphy3.MorphAnalyzer()
NORMALIZED_CACHE = {}

def normalize_word(word: str) -> str:
    """Приводит слово к нормальной форме с помощью pymorphy3"""

    if word in NORMALIZED_CACHE:
        return NORMALIZED_CACHE[word]
    try:
        stemmed = stemmer.stem(word)

        if stemmed in KEYWORDS_NORMALIZED:
            NORMALIZED_CACHE[word] = stemmed
            return stemmed
        
        parsed_variants = morph.parse(word)
        for variant in parsed_variants[:5]: 
            normalized = variant.normal_form
            if normalized in KEYWORDS_NORMALIZED:
                NORMALIZED_CACHE[word] = normalized
                return normalized
            
        if parsed_variants:
            best_variant = parsed_variants[0]
            normalized = best_variant.normal_form
            NORMALIZED_CACHE[word] = normalized
            return normalized
    except:
        normalized = word.lower()
        NORMALIZED_CACHE[word] = normalized
        return normalized


class TokenType(Enum):
    COMMAND_FIND = "найди"
    COMMAND_DELETE = "удали"
    COMMAND_OUTPUT = "выведи"

    PRODUCT_CAKE = "торт"
    PRODUCT_PASTRY = "пирожное"
    PRODUCT_DESSERT = "десерт"

    WITH = "со"  
    TASTE = "вкус"
    WITH_FILLING = "с" 
    FILLING = "начинка"

    FOR = "за"
    AFTER = "после"
    BEFORE = "до"
    KEYWORD_NUMBER = "число"
    KEYWORD_NUMBERS = "числа"
    CHEAPER = "дешевле"
    EXPENSIVE = "дороже"
    COST = "стоимостью"
    WITHOUT = "без"

    CURRENCY_RUB = "рубль"
    CURRENCY_THOUSAND = "тысяча"

    AND = "и"

    DOT = "."
    NUMBER = "NUMBER"
    DAY = "DAY"
    MONTH = "MONTH"
    YEAR = "YEAR"

    LEXEME = "LEXEME"

    EOF = "EOF"

KEYWORDS = {
    "найди": TokenType.COMMAND_FIND,
    "удали": TokenType.COMMAND_DELETE,
    "выведи": TokenType.COMMAND_OUTPUT,
    "торт": TokenType.PRODUCT_CAKE,
    "пирожное": TokenType.PRODUCT_PASTRY,
    "десерт": TokenType.PRODUCT_DESSERT,
    "со": TokenType.WITH,
    "вкус": TokenType.TASTE,
    "с": TokenType.WITH_FILLING,
    "начинка": TokenType.FILLING,
    "за": TokenType.FOR,
    "после": TokenType.AFTER,
    "до": TokenType.BEFORE,
    "число": TokenType.KEYWORD_NUMBER,
    "числа": TokenType.KEYWORD_NUMBERS,
    "дешевле": TokenType.CHEAPER,
    "дороже": TokenType.EXPENSIVE,
    "стоимостью": TokenType.COST,
    "без": TokenType.WITHOUT,
    "рубль": TokenType.CURRENCY_RUB,
    "тысяча": TokenType.CURRENCY_THOUSAND,
    "и": TokenType.AND,
}

KEYWORDS_NORMALIZED = set(KEYWORDS.keys())

VALID_DAYS = {f"{i:02d}" for i in range(1, 32)} | {str(i) for i in range(1, 32)}
VALID_MONTHS = {f"{i:02d}" for i in range(1, 13)} | {str(i) for i in range(1, 13)}
VALID_YEARS = {"2025", "2026"}


@dataclass
class Token:
    type: TokenType
    value: str
    original: str
    position: int
    line: int = 1
    
    def __repr__(self):
        return f"Token({self.type.name}, '{self.value}')"


class Lexer:
    """Лексер с предварительной очисткой текста"""
    
    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        self.line = 1
        self.tokens = []
    
    def tokenize(self) -> List[Token]:
        """Разбивает текст на токены"""

        self.tokens = []
        lines = self.text.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            self._process_line(line, line_num)
        self.tokens.append(Token(TokenType.EOF, '', '', self.pos, self.line))
        return self.tokens
    
    def _process_line(self, line: str, line_num: int):
        """Обрабатывает одну строку текста"""

        i = 0
        n = len(line)
        
        while i < n:
            if line[i].isspace():
                i += 1
                continue
            if line[i] in ',!?:;()[]{}"\'—–-':
                i += 1
                continue

            if line[i].isdigit():
                start = i
                has_dot = False
                
                while i < n and (line[i].isdigit() or line[i] == '.'):
                    if line[i] == '.':
                        has_dot = True
                    i += 1
                
                token_str = line[start:i]
                
                if has_dot:
                    parts = token_str.split('.')
                    if len(parts) == 3:
                        self.tokens.append(Token(TokenType.DAY, parts[0], parts[0], start, line_num))
                        self.tokens.append(Token(TokenType.DOT, '.', '.', start, line_num))
                        self.tokens.append(Token(TokenType.MONTH, parts[1], parts[1], start, line_num))
                        self.tokens.append(Token(TokenType.DOT, '.', '.', start, line_num))
                        self.tokens.append(Token(TokenType.YEAR, parts[2], parts[2], start, line_num))
                    else:
                        self.tokens.append(Token(TokenType.NUMBER, token_str, token_str, start, line_num))
                else:
                    self.tokens.append(Token(TokenType.NUMBER, token_str, token_str, start, line_num))
                
                continue
            if line[i].isalpha() or line[i] == '-':
                start = i
                while i < n and (line[i].isalpha() or line[i] == '-'):
                    i += 1
                
                word = line[start:i]
                word_lower = word.lower() 

                if word_lower in KEYWORDS_NORMALIZED:
                    self.tokens.append(Token(KEYWORDS[word_lower], word_lower, word, start, line_num))
                    continue

                normalized = normalize_word(word_lower)

                if normalized in KEYWORDS_NORMALIZED:
                    self.tokens.append(Token(KEYWORDS[normalized], normalized, word, start, line_num))
                else:
                    self.tokens.append(Token(TokenType.LEXEME, word_lower, word, start, line_num))
                
                continue
            
            i += 1

class ParseError(Exception):
    def __init__(self, message: str, token: Optional[Token] = None):
        self.message = message
        self.token = token
        super().__init__(message)
    
    def __str__(self):
        if self.token:
            return f"Ошибка на позиции {self.token.position} (строка {self.token.line}): {self.message}, найден '{self.token.original}'"
        return self.message


class Node:
    def __init__(self, type: str, value: Any = None, original: str = None):
        self.type = type
        self.value = value
        self.original = original
        self.children = []
    
    def add_child(self, child: 'Node'):
        self.children.append(child)
    
    def __repr__(self, level=0, prefix="├─ "):
        ret = "  " * level + prefix + f"{self.type}"
        if self.value is not None:
            ret += f": {self.value}"
        if self.original and self.original != self.value:
            ret += f" (ориг: {self.original})"
        ret += "\n"
        for i, child in enumerate(self.children):
            is_last = (i == len(self.children) - 1)
            ret += child.__repr__(level + 1, "└─ " if is_last else "├─ ")
        return ret


class Parser:
    """LL(1) парсер для грамматики DSL Кондитерская"""
    
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0
        self.current_token = tokens[0] if tokens else None
    
    def advance(self):
        self.pos += 1
        if self.pos < len(self.tokens):
            self.current_token = self.tokens[self.pos]
        else:
            self.current_token = None
    
    def match(self, *expected_types: TokenType) -> Optional[Token]:
        if self.current_token and self.current_token.type in expected_types:
            token = self.current_token
            self.advance()
            return token
        return None
    
    def expect(self, *expected_types: TokenType) -> Token:
        token = self.match(*expected_types)
        if not token:
            if self.current_token:
                expected_names = [t.name for t in expected_types]
                raise ParseError(
                    f"Ожидался один из: {', '.join(expected_names)}", 
                    self.current_token
                )
            else:
                raise ParseError("Неожиданный конец ввода")
        return token
    
    def parse(self) -> Node:
        try:
            return self.parse_query()
        except ParseError as e:
            error_node = Node("ERROR")
            error_node.value = str(e)
            return error_node
    
    def parse_query(self) -> Node:
        node = Node("QUERY")
        
        cmd_node = self.parse_command()
        node.add_child(cmd_node)
        
        self._parse_query_part(node)
        
        while True:
            conj_node = self.parse_conjunction()
            if conj_node:
                node.add_child(conj_node)
            else:
                break
        
        if self.current_token and self.current_token.type != TokenType.EOF:
            raise ParseError("Неожиданный токен после завершения запроса", self.current_token)
        
        return node

    def _parse_query_part(self, node: Node):
        obj_node = self.parse_object()
        node.add_child(obj_node)
        
        taste_node = self.parse_taste()
        if taste_node:
            node.add_child(taste_node)
        
        filling_node = self.parse_filling()
        if filling_node:
            node.add_child(filling_node)
        
        filters_node = self.parse_filters()
        if filters_node:
            node.add_child(filters_node)
    
    def parse_command(self) -> Node:
        token = self.expect(
            TokenType.COMMAND_FIND,
            TokenType.COMMAND_DELETE,
            TokenType.COMMAND_OUTPUT
        )
        return Node("COMMAND", token.value, token.original)
    
    def parse_object(self) -> Node:
        node = Node("OBJECT")
        
        token = self.expect(
            TokenType.PRODUCT_CAKE,
            TokenType.PRODUCT_PASTRY,
            TokenType.PRODUCT_DESSERT
        )
        node.add_child(Node("PRODUCT_TYPE", token.value, token.original))
        
        if self.current_token and self.current_token.type == TokenType.LEXEME:
            name_token = self.expect(TokenType.LEXEME)
            node.add_child(Node("NAME", name_token.value, name_token.original))
        
        return node
    
    def parse_taste(self) -> Optional[Node]:
        if self.current_token and self.current_token.type == TokenType.WITH:
            saved_pos = self.pos
            saved_token = self.current_token
            
            try:
                node = Node("TASTE")

                with_token = self.expect(TokenType.WITH)
                node.add_child(Node("WITH", with_token.value, with_token.original))
                
                taste_token = self.expect(TokenType.TASTE)
                node.add_child(Node("TASTE_KEYWORD", taste_token.value, taste_token.original))

                taste_list_node = self.parse_taste_list()
                node.add_child(taste_list_node)
                
                return node
            except ParseError:
                self.pos = saved_pos
                self.current_token = saved_token
                return None
        return None

    
    def parse_taste_list(self) -> Node:
        node = Node("TASTE_LIST")
        
        token = self.expect(TokenType.LEXEME)
        node.add_child(Node("TASTE_ITEM", token.value, token.original))
        
        while self.current_token and self.current_token.type == TokenType.LEXEME:
            token = self.expect(TokenType.LEXEME)
            node.add_child(Node("TASTE_ITEM", token.value, token.original))
        
        return node
    
    def parse_filling(self) -> Optional[Node]:
        if self.current_token and self.current_token.type == TokenType.WITH_FILLING:
            saved_pos = self.pos
            saved_token = self.current_token
            
            try:
                node = Node("FILLING")

                with_token = self.expect(TokenType.WITH_FILLING)
                node.add_child(Node("WITH_FILLING", with_token.value, with_token.original))

                filling_token = self.expect(TokenType.FILLING)
                node.add_child(Node("FILLING_KEYWORD", filling_token.value, filling_token.original))

                filling_list_node = self.parse_filling_list()
                node.add_child(filling_list_node)
                
                return node
            except ParseError:
                self.pos = saved_pos
                self.current_token = saved_token
                return None
        return None
    
    def parse_filling_list(self) -> Node:
        node = Node("FILLING_LIST")
        
        token = self.expect(TokenType.LEXEME)
        node.add_child(Node("FILLING_ITEM", token.value, token.original))
        
        while self.current_token and self.current_token.type == TokenType.LEXEME:
            token = self.expect(TokenType.LEXEME)
            node.add_child(Node("FILLING_ITEM", token.value, token.original))
        
        return node
    
    def parse_filters(self) -> Optional[Node]:
        if self.current_token and self.current_token.type in [
            TokenType.FOR, TokenType.AFTER, TokenType.BEFORE,
            TokenType.CHEAPER, TokenType.EXPENSIVE, TokenType.COST,
            TokenType.WITHOUT
        ]:
            node = Node("FILTERS")
            
            filter_node = self.parse_filter()
            node.add_child(filter_node)
            
            while self.current_token and self.current_token.type in [
                TokenType.FOR, TokenType.AFTER, TokenType.BEFORE,
                TokenType.CHEAPER, TokenType.EXPENSIVE, TokenType.COST,
                TokenType.WITHOUT
            ]:
                filter_node = self.parse_filter()
                node.add_child(filter_node)
            
            return node
        return None
    
    def parse_filter(self) -> Node:
        if not self.current_token:
            raise ParseError("Неожиданный конец ввода при разборе фильтра")
        
        if self.current_token.type in [TokenType.FOR, TokenType.AFTER, TokenType.BEFORE]:
            return self.parse_number_filter()
        elif self.current_token.type in [TokenType.CHEAPER, TokenType.EXPENSIVE, TokenType.COST]:
            return self.parse_price_filter()
        elif self.current_token.type == TokenType.WITHOUT:
            return self.parse_ingredient_filter()
        else:
            raise ParseError("Неожиданный токен в фильтре", self.current_token)
    
    def parse_number_filter(self) -> Node:
        node = Node("NUMBER_FILTER")
        
        prefix_token = self.expect(TokenType.FOR, TokenType.AFTER, TokenType.BEFORE)
        node.add_child(Node("PREFIX", prefix_token.value, prefix_token.original))
        
        date_node = Node("DATE")
        
        day_token = self.expect(TokenType.DAY)
        date_node.add_child(Node("DAY", day_token.value, day_token.original))
        
        self.expect(TokenType.DOT)
        
        month_token = self.expect(TokenType.MONTH)
        date_node.add_child(Node("MONTH", month_token.value, month_token.original))
        
        self.expect(TokenType.DOT)
        
        year_token = self.expect(TokenType.YEAR)
        date_node.add_child(Node("YEAR", year_token.value, year_token.original))
        
        node.add_child(date_node)
        
        if prefix_token.type == TokenType.FOR:
            keyword_token = self.expect(TokenType.KEYWORD_NUMBER)
            node.add_child(Node("KEYWORD", keyword_token.value, keyword_token.original))
        else:
            keyword_token = self.expect(TokenType.KEYWORD_NUMBERS)
            node.add_child(Node("KEYWORD", keyword_token.value, keyword_token.original))
        
        return node
    
    def parse_price_filter(self) -> Node:
        node = Node("PRICE_FILTER")
        
        prefix_token = self.expect(TokenType.CHEAPER, TokenType.EXPENSIVE, TokenType.COST)
        node.add_child(Node("PREFIX", prefix_token.value, prefix_token.original))
        
        number_token = self.expect(TokenType.NUMBER, TokenType.DAY, TokenType.MONTH, TokenType.YEAR)
        node.add_child(Node("NUMBER", number_token.value, number_token.original))
        
        currency_token = self.expect(TokenType.CURRENCY_RUB, TokenType.CURRENCY_THOUSAND)
        node.add_child(Node("CURRENCY", currency_token.value, currency_token.original))
        
        return node
    
    def parse_ingredient_filter(self) -> Node:
        node = Node("INGREDIENT_FILTER")

        without_token = self.expect(TokenType.WITHOUT)
        node.add_child(Node("WITHOUT", without_token.value, without_token.original))

        ingredients = Node("INGREDIENT_LIST")
        
        token = self.expect(TokenType.LEXEME)
        ingredients.add_child(Node("INGREDIENT", token.value, token.original))
        
        while self.current_token and self.current_token.type == TokenType.LEXEME:
            token = self.expect(TokenType.LEXEME)
            ingredients.add_child(Node("INGREDIENT", token.value, token.original))
        
        node.add_child(ingredients)
        return node
    
    def parse_conjunction(self) -> Optional[Node]:
        if not self.current_token:
            return None
        
        if self.current_token.type == TokenType.AND or \
           self.current_token.type in [
               TokenType.PRODUCT_CAKE,
               TokenType.PRODUCT_PASTRY,
               TokenType.PRODUCT_DESSERT
           ]:
            node = Node("CONJUNCTION")
            
            if self.current_token.type == TokenType.AND:
                and_token = self.expect(TokenType.AND)
                node.add_child(Node("CONJ_WORD", "и"))
            
            subquery_node = self.parse_query_without_command()
            node.add_child(subquery_node)
            
            return node
        
        return None
    
    def parse_query_without_command(self) -> Node:
        node = Node("SUBQUERY")
        
        obj_node = self.parse_object()
        node.add_child(obj_node)
        
        taste_node = self.parse_taste()
        if taste_node:
            node.add_child(taste_node)
        
        filling_node = self.parse_filling()
        if filling_node:
            node.add_child(filling_node)
        
        filters_node = self.parse_filters()
        if filters_node:
            node.add_child(filters_node)
        
        return node

def visualize_tree(node: Node, filename):
    """Визуализирует дерево разбора с помощью graphviz"""

    dot = graphviz.Digraph(comment='Parse Tree', format='png')
    dot.attr('node', shape='box', style='rounded', fontname='Arial')
    
    def add_nodes(parent_node, parent_id=None):
        node_id = str(id(parent_node))
        label = parent_node.type
        if parent_node.value:
            label += f"\n{parent_node.value}"
        if parent_node.original and parent_node.original != parent_node.value:
            label += f"\n(ориг: {parent_node.original})"
        dot.node(node_id, label)
        if parent_id:
            dot.edge(parent_id, node_id)
        for child in parent_node.children:
            add_nodes(child, node_id)
    
    add_nodes(node)
    dot.render(filename, view=True, cleanup=False)
    print(f"Граф сохранен как {filename}.png")

def process_queries(filename: str):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            queries = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Файл {filename} не найден.")
        return
    
    for i, query in enumerate(queries, 1):
        print(f"ЗАПРОС {i}: \"{query}\"")
        print()
        
        try:
            lexer = Lexer(query)
            tokens = lexer.tokenize()
            
            parser = Parser(tokens)
            parse_tree = parser.parse()
            
            if parse_tree.type == "ERROR":
                print("РЕЗУЛЬТАТ: НЕУДАЧА")
                print(f"ДИАГНОСТИКА: {parse_tree.value}")
            else:
                print("РЕЗУЛЬТАТ: УСПЕХ")
                print("\nДЕРЕВО РАЗБОРА:")
                print(parse_tree)
                visualize_tree(parse_tree, filename)
                
        except Exception as e:
            print("РЕЗУЛЬТАТ: НЕУДАЧА (исключение)")
            print(f"ДИАГНОСТИКА: {e}")


def create_test_file():
    test_queries = [
        "Принеси торты",
        "Найди торт Панчо, выведи торт Наполеон",
        "Удали десерт со вкусом шоколад и вишня",
        "Удали торт стоимостью 5 долларов",
        "Выведи пирожные за 5 число"
    ]
    
    with open("queries.txt", "w", encoding="utf-8") as f:
        for q in test_queries:
            f.write(q + "\n")


if __name__ == "__main__":
    while True:
        print("Введите test для тестов или text для ввода запроса, exit для выхода")
        action = input()
        if action == "test":
            create_test_file()
            process_queries("queries.txt")
        elif action == "text":
            with open("queries.txt", "w", encoding="utf-8") as f:
                f.write(input())
            process_queries("queries.txt")
            
        elif action == "exit":
            break
        else: 
            print("Проверьте ввод")
